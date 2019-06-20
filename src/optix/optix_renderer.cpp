#include "openmc/error.h"

#include "openmc/optix/optix_renderer.h"
#include "openmc/optix/tetrahedron.h"

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "cuda/common.h"
#include "cuda/helpers.h"
#include "sutil.h"

namespace openmc {

using namespace optix;

void OptiXRenderer::render() {
  write_message("Rendering geometry...", 5);

  createContext();
  createMaterials();
  setupScene();
  setupCamera();
  write_message("Launching OptiX context...", 5);
  context->validate();
  updateCamera();
  context->launch( 0, width, height );
  write_message("Writing output file...", 5);
  sutil::displayBufferPPM( "test.ppm", context[ "output_buffer" ]->getBuffer() );
  write_message("Destroying OptiX context...", 5);
  context->destroy();

}

void OptiXRenderer::createContext() {
  write_message("Creating OptiX context...", 5);

#if OPTIX_VERSION / 10000 >= 6
  int rtx = 1;
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx) != RT_SUCCESS) {
    printf("Error setting RT_GLOBAL_ATTRIBUTE_ENABLE_RTX!\n");
  } else {
    printf("RTX execution mode is %s\n", (rtx) ? "ON" : "OFF");
  }
#endif

  // Set up context
  context = Context::create();
  context->setRayTypeCount(2);
  context->setEntryPointCount(1);
  context->setStackSize(2800);
#if OPTIX_VERSION / 10000 > 6
  context->setMaxTraceDepth( 12 );
#endif
  // Note: high max depth for reflection and refraction through glass
  context["max_depth"]->setInt(10);
  context["frame"]->setUint(0u);
  context["scene_epsilon"]->setFloat(1.e-4f);
  context["ambient_light_color"]->setFloat(0.4f, 0.4f, 0.4f);

  bool use_pbo = false;
  Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
  context["output_buffer"]->set(buffer);

  // Accumulation buffer.
  Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                              RT_FORMAT_FLOAT4, width, height);
  context["accum_buffer"]->set(accum_buffer);

  // Ray generation program
  const char *ptx = sutil::getPtxString(SAMPLE_NAME, "accum_camera.cu");
  Program ray_gen_program = context->createProgramFromPTXString(ptx, "pinhole_camera");
  context->setRayGenerationProgram(0, ray_gen_program);

  // Exception program
  Program exception_program = context->createProgramFromPTXString(ptx, "exception");
  context->setExceptionProgram(0, exception_program);
  context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

  // Miss program
  ptx = sutil::getPtxString(SAMPLE_NAME, "constantbg.cu");
  context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));
  context["bg_color"]->setFloat(0.34f, 0.55f, 0.85f);
}

void OptiXRenderer::setupCamera() {
  camera_eye = make_float3(-1.0f, 1.0f, 10.0f);
  camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
  camera_up = make_float3(0.0f, 1.0f, 0.0f);
  camera_rotate = Matrix4x4::identity();
}

void OptiXRenderer::updateCamera() {
  const float vfov = 60.0f;
  const float aspect_ratio = static_cast<float>(width) /
                             static_cast<float>(height);

  float3 camera_u, camera_v, camera_w;
  sutil::calculateCameraVariables(
    camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
    camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

  const Matrix4x4 frame = Matrix4x4::fromBasis(
    normalize(camera_u),
    normalize(camera_v),
    normalize(-camera_w),
    camera_lookat);
  const Matrix4x4 frame_inv = frame.inverse();
  // Apply camera rotation twice to match old SDK behavior
  const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

  camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
  camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
  camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

  sutil::calculateCameraVariables(
    camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
    camera_u, camera_v, camera_w, true);

  camera_rotate = Matrix4x4::identity();

  context["eye"]->setFloat(camera_eye);
  context["U"]->setFloat(camera_u);
  context["V"]->setFloat(camera_v);
  context["W"]->setFloat(camera_w);
}

void OptiXRenderer::createMaterials() {
  // Normal shader material
  const char *ptx = sutil::getPtxString(SAMPLE_NAME, "normal_shader.cu");
  Program tri_ch = context->createProgramFromPTXString(ptx, "closest_hit_radiance");
  Program tri_ah = context->createProgramFromPTXString(ptx, "any_hit_shadow");

  material = context->createMaterial();
  material->setClosestHitProgram(0, tri_ch);
  material->setAnyHitProgram(1, tri_ah);
}


GeometryGroup OptiXRenderer::createGeometryTriangles() {
  // Create a tetrahedron using four triangular faces.  First We will create
  // vertex and index buffers for the faces, and then create a
  // GeometryTriangles object.
  const unsigned num_faces = 4;
  const unsigned num_vertices = num_faces * 3;

  // Define a regular tetrahedron of height 2, translated 1.5 units from the origin.
  Tetrahedron tet(5.0f, make_float3(0.0f, -5.0f, 0.0f));
  // Tetrahedron tet(20.0f, make_float3(0.0f, -5.0f, 0.0f));

  // Create Buffers for the triangle vertices, normals, texture coordinates, and indices.
  Buffer vertex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices);
  Buffer normal_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices);
  Buffer texcoord_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_vertices);
  Buffer index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, num_faces);

  // Copy the tetrahedron geometry into the device Buffers.
  memcpy(vertex_buffer->map(), tet.vertices, sizeof(tet.vertices));
  memcpy(normal_buffer->map(), tet.normals, sizeof(tet.normals));
  memcpy(texcoord_buffer->map(), tet.texcoords, sizeof(tet.texcoords));
  memcpy(index_buffer->map(), tet.indices, sizeof(tet.indices));

  vertex_buffer->unmap();
  normal_buffer->unmap();
  texcoord_buffer->unmap();
  index_buffer->unmap();

#if OPTIX_VERSION / 10000 >= 6
  // Create a GeometryTriangles object.
  optix::GeometryTriangles geom_tri = context->createGeometryTriangles();

  geom_tri->setPrimitiveCount( num_faces );
  geom_tri->setTriangleIndices( index_buffer, RT_FORMAT_UNSIGNED_INT3 );
  geom_tri->setVertices( num_vertices, vertex_buffer, RT_FORMAT_FLOAT3 );
  geom_tri->setBuildFlags( RTgeometrybuildflags( 0 ) );

  // Set an attribute program for the GeometryTriangles, which will compute
  // things like normals and texture coordinates based on the barycentric
  // coordindates of the intersection.
  const char* ptx = sutil::getPtxString( SAMPLE_NAME, "optixGeometryTriangles.cu" );
  geom_tri->setAttributeProgram( context->createProgramFromPTXString( ptx, "triangle_attributes" ) );

  geom_tri["index_buffer"]->setBuffer( index_buffer );
  geom_tri["vertex_buffer"]->setBuffer( vertex_buffer );
  geom_tri["normal_buffer"]->setBuffer( normal_buffer );
  geom_tri["texcoord_buffer"]->setBuffer( texcoord_buffer );

  // Bind a Material to the GeometryTriangles.  Materials can be shared
  // between GeometryTriangles objects and other Geometry types, as long as
  // all of the attributes needed by the attached hit programs are produced in
  // the attribute program.
  tri_gi = context->createGeometryInstance( geom_tri, material );
#else
  optix::Geometry geom_tri = context->createGeometry();
  tri_gi = context->createGeometryInstance(geom_tri, &material, &material + 1);
#endif
  GeometryGroup tri_gg = context->createGeometryGroup();
  tri_gg->addChild(tri_gi);
  tri_gg->setAcceleration(context->createAcceleration("Trbvh"));

  return tri_gg;
}

void OptiXRenderer::setupScene() {
  GeometryGroup tri_gg = createGeometryTriangles();
  context["top_object"]->set(tri_gg);
  context["top_shadower"]->set(tri_gg);
}

}