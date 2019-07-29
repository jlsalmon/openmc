import glob
import os
import re
import subprocess
import sys
import pandas as pd
import time


def main():
    experiments = [
        {
            "name": "cube",
            "versions": [
                {"name": "optix", "host": "turing", "rtx": True},
                {"name": "optix", "host": "turing", "rtx": False},
                {"name": "optix", "host": "gtx1080", "rtx": True},
                {"name": "optix", "host": "gtx1080", "rtx": False},
                {"name": "native", "host": "turing", "rtx": False},
                {"name": "dagmc", "host": "turing", "rtx": False}
            ]
        },
        {
            "name": "sphere",
            "versions": [
                {"name": "optix", "host": "turing", "rtx": True},
                {"name": "optix", "host": "turing", "rtx": False},
                {"name": "optix", "host": "gtx1080", "rtx": True},
                {"name": "optix", "host": "gtx1080", "rtx": False},
                {"name": "native", "host": "turing", "rtx": False},
                {"name": "dagmc", "host": "turing", "rtx": False}
            ]
        },
        {
            "name": "buddha",
            "versions": [
                {"name": "optix", "host": "turing", "rtx": True},
                {"name": "optix", "host": "turing", "rtx": False},
                {"name": "optix", "host": "gtx1080", "rtx": True},
                {"name": "optix", "host": "gtx1080", "rtx": False},
            ]
        },
        {
            "name": "dragon",
            "versions": [
                {"name": "optix", "host": "turing", "rtx": True},
                {"name": "optix", "host": "turing", "rtx": False},
                {"name": "optix", "host": "gtx1080", "rtx": True},
                {"name": "optix", "host": "gtx1080", "rtx": False},
            ]
        },
        {
            "name": "hairball",
            "versions": [
                {"name": "optix", "host": "turing", "rtx": True},
                {"name": "optix", "host": "turing", "rtx": False},
                {"name": "optix", "host": "gtx1080", "rtx": True},
                {"name": "optix", "host": "gtx1080", "rtx": False},
            ]
        }
    ]

    particle_sizes = [1000, 10000, 100000, 1000000, 10000000]
    num_runs = 5
    resume = True

    if resume:
        # Load the most recent file
        filename = max(glob.glob('experiments/results_*.tsv'), key=os.path.getctime)
        results = pd.read_csv(filename, sep='\t', index_col=0)
        print(f"Resuming from most recent run, {filename}")
    else:
        # Determine the next filename
        i = 0
        while os.path.exists(f"experiments/results_{i:04d}.tsv"):
            i += 1
        filename = f"experiments/results_{i:04d}.tsv"
        results = pd.DataFrame(columns=[
            "experiment",
            "version",
            "host",
            "rtx",
            "num_particles",
            "calculation_rate",
            "time_in_transport",
        ])

    # # Ensure the remote sources are synced
    # for host in hosts:
    #     cmd = f"rsync -avP ./ {host}:openmc --exclude cmake-build-debug --exclude build \
    #                   --exclude .git --exclude endf71_hdf5 --exclude '*.ppm' --exclude openmc/capi/libopenmc.dylib \
    #                   --exclude *.blend* --exclude *.trelis* --exclude *.stl --exclude geometry-no-boundary.obj \
    #                   --exclude '*.h5' --exclude '*.vti' --exclude '*.vtk' --exclude '*.stl'"
    #     run(cmd)
    #     print(f"Synced sources on {host}")
    #
    # # Compile the remote sources
    # for host in hosts:
    #     run(f"ssh {host} -t 'make -j 16 --directory=openmc/build'")
    #     print(f"Compiled sources on {host}")

    # Run each model three times and take the average
    for experiment in experiments:
        for version in experiment["versions"]:
            for num_particles in particle_sizes:

                # Check if we already have this result
                if have_result(results, experiment, version, num_particles):
                    print(f"Skipping {experiment['name']}/{version['name']} "
                          f"on {version['host']}, rtx={version['rtx']} "
                          f"with n={num_particles}")
                    continue

                # Skip the super slow DAGMC runs
                if version["name"] == "dagmc" and num_particles == particle_sizes[-1]:
                    continue

                calculation_rate_sum = time_in_transport_sum = 0.

                # Skip the first run
                for i in range(num_runs + 1):

                    cmd = f"cd experiments/{experiment['name']}/{version['name']} " \
                        f"&& ~/.local/bin/openmc --particles {num_particles}"
                    if not version["rtx"]:
                        cmd += " --no-triangle-api --no-rtx"

                    if version["host"] == "gtx1080":
                        cmd = f"ssh wr18313@node21 -t 'cd openmc && " + cmd + "'"

                    while True:
                        try:
                            print(f"Running {experiment['name']}/{version['name']} "
                                  f"on {version['host']}, rtx={version['rtx']} "
                                  f"with n={num_particles}")
                            output = run(cmd)
                            calculation_rate, time_in_transport = extract_results(output)
                            break
                        except AttributeError:
                            print("Retrying...")

                    # Skip the first run
                    if i > 0:
                        calculation_rate_sum += calculation_rate
                        time_in_transport_sum += time_in_transport

                avg_calculation_rate = calculation_rate_sum / num_runs
                avg_time_in_transport = time_in_transport_sum / num_runs

                print(f"{experiment['name']}/{version['name']} {version['host']} "
                      f"n={num_particles} rtx={version['rtx']} "
                      f"calculation_rate={avg_calculation_rate} "
                      f"time_in_transport={avg_time_in_transport}")

                # Insert this result
                results.loc[len(results)] = [
                    experiment['name'], version['name'], version['host'],
                    version['rtx'], num_particles,
                    avg_calculation_rate, avg_time_in_transport
                ]

                # Periodically save the results
                results.to_csv(filename, sep='\t', encoding='utf-8')

    # TODO: Plot the results


def have_result(df, experiment, version, num_particles):
    return (
            (df['experiment'] == experiment['name']) &
            (df['version'] == version['name']) &
            (df['host'] == version['host']) &
            (df['rtx'] == version['rtx']) &
            (df['num_particles'] == num_particles) &
            (df['calculation_rate'] > 0) &
            (df['time_in_transport'] > 0)
    ).any()


def extract_results(output):
    m = re.search("Time in transport only {10}= (?P<time_in_transport>.*) seconds", output)
    time_in_transport = float(m.group("time_in_transport"))
    m = re.search("Calculation Rate .active. {9}= (?P<calculation_rate>.*) particles/second", output)
    calculation_rate = float(m.group("calculation_rate"))
    return calculation_rate, time_in_transport


def run(cmd, write_to_stdout=False):
    print(cmd)
    output = []
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    while not process.poll():
        line = process.stdout.readline().decode()
        if line:
            if write_to_stdout:
                sys.stdout.write(line)
            output.append(line)
        else:
            break
    output = "".join(output)
    print(output)
    return output


if __name__ == '__main__':
    main()
