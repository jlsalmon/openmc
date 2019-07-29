import glob
import os
import subprocess

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Fetch the results
    subprocess.run("scp turing:openmc/experiments/results* experiments/", shell=True)

    # Load the most recent results file
    files = glob.glob('experiments/results_*.tsv')
    latest = max(files, key=os.path.getctime)
    df = pd.read_csv(latest, sep='\t', index_col=0)

    # Calculation rate for cube model
    calculation_rate_vs_num_particles_bar(df, "cube", "Cube")
    time_in_transport_vs_num_particles_scatter(df, "cube", "Cube")

    calculation_rate_vs_num_particles_bar(df, "sphere", "Sphere")
    time_in_transport_vs_num_particles_scatter(df, "sphere", "Sphere")

    calculation_rate_vs_num_particles_bar(df, "buddha", "Happy Buddha")
    time_in_transport_vs_num_particles_scatter(df, "buddha", "Happy Buddha")

    calculation_rate_vs_num_particles_bar(df, "dragon", "Asian Dragon")
    time_in_transport_vs_num_particles_scatter(df, "dragon", "Asian Dragon")

    calculation_rate_vs_num_particles_bar(df, "hairball", "Hairball")
    time_in_transport_vs_num_particles_scatter(df, "hairball", "Hairball")


def calculation_rate_vs_num_particles_bar(df, experiment, model_name):
    results = df.loc[(df['experiment'] == experiment)]

    df = pd.pivot_table(results,
                        index=["version", "host", "rtx"],
                        columns="num_particles",
                        values="calculation_rate").T

    ax = df.plot.bar(rot=0)

    # Customise legend
    ax.legend(get_legend(list(df)))

    ax.set_xticklabels([r"{:.0e}".format(label) for label in df.index.tolist()])
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Calculation Rate (particles/sec)")
    ax.set_title(f"Model: {model_name}")

    plt.show()


def time_in_transport_vs_num_particles_scatter(df, experiment, model_name):
    results = df.loc[(df['experiment'] == experiment)]

    df = pd.pivot_table(results,
                        index=["version", "host", "rtx"],
                        columns="num_particles",
                        values="time_in_transport").T

    ax = df.plot(marker='o', markersize=6, linestyle='--', linewidth=1)

    # Customise legend
    ax.legend(get_legend(list(df)))

    ax.set_xticklabels([r"{:.0e}".format(label) for label in df.index.tolist()])
    ax.set_xlabel("Number of Particles")
    ax.set_xscale('log')
    ax.set_ylabel("Time in Transport (ms)")
    ax.set_yscale('log')
    ax.set_title(f"Model: {model_name}")

    plt.show()


def get_legend(groups):
    legend = []
    for group in groups:
        if group == ('dagmc', 'turing', False):
            legend.append("DAGMC")
        if group == ('native', 'turing', False):
            legend.append("OpenMC Native")
        if group == ('optix', 'gtx1080', False):
            legend.append("OptiX, GTX 1080 Ti, RTX=OFF")
        if group == ('optix', 'gtx1080', True):
            legend.append("OptiX, GTX 1080 Ti, RTX=ON")
        if group == ('optix', 'turing', False):
            legend.append("OptiX, RTX 2080 Ti, RTX=OFF")
        if group == ('optix', 'turing', True):
            legend.append("OptiX, RTX 2080 Ti, RTX=ON")
    return legend


if __name__ == '__main__':
    plt.style.use('seaborn')
    main()
