#ifndef OPENMC_UTILS_H
#define OPENMC_UTILS_H

std::string directoryOfFilePath(const std::string &filepath) {
  size_t slash_pos, backslash_pos;
  slash_pos = filepath.find_last_of('/');
  backslash_pos = filepath.find_last_of('\\');

  size_t break_pos;
  if (slash_pos == std::string::npos && backslash_pos == std::string::npos) {
    return std::string();
  } else if (slash_pos == std::string::npos) {
    break_pos = backslash_pos;
  } else if (backslash_pos == std::string::npos) {
    break_pos = slash_pos;
  } else {
    break_pos = std::max(slash_pos, backslash_pos);
  }

  // Include the final slash
  return filepath.substr(0, break_pos + 1);
}


// std::string getExtension(const std::string &filename) {
//   // Get the filename extension
//   std::string::size_type extension_index = filename.find_last_of(".");
//   std::string ext = extension_index != std::string::npos ?
//                     filename.substr(extension_index + 1) :
//                     std::string();
//   std::locale loc;
//   for (std::string::size_type i = 0; i < ext.length(); ++i)
//     ext[i] = std::tolower(ext[i], loc);
//
//   return ext;
// }

#endif //OPENMC_UTILS_H
