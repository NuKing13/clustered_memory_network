from datetime import time
from seaborn.matrix import heatmap
from sys import argv
import signals as sg
import numpy as np
import pickle
import json
import os
import argparse
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlibsettings as pltst
import capacities as cap
import signals as sg
import json
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.patches as mpatches

colours = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def load_results(
    folder, data_type="json", trans_cap_data=False
):  # , lable = {'r_bg' : [], 'g' : [], 'marker': ''}):
    results = [] #dict list
    for path in os.listdir(folder):
        fi = os.path.join(folder, path)
        if data_type == "json":
            with open(fi, "rb") as file:
                try:
                    tmp = json.load(file)
                    results.append(tmp)
                except Exception as Error:
                    print(Error)
        elif data_type == "p" or data_type == "pickle":
            with open(fi, "rb") as file:
                try:
                    tmp = pickle.load(file)
                    if trans_cap_data == True:
                        tmp["vec"] = cap2vec(tmp["allcaps"])
                    results.append(tmp)
                except Exception as Error:
                    print(Error)
    return results


def data_matrix(folder, index, columns, values):
    results = pd.DataFrame(load_results(folder))
    pivot = results.pivot(index=index, columns=columns, values=values)
    a, b, c = pivot.to_numpy(), pivot.index, pivot.columns
    return pivot


def set_lables(results, lables={"rg_tuples": [], "marker": []}):
    for result in results:
        if (result["r_bg"], result["g"]) in lables["rg_tuples"]:
            result["lable"] = lables["marker"][
                lables["rg_tuples"].index((result["r_bg"], result["g"]))
            ]
        else:
            result["lable"] = ""
    return results


def set_lables_cmod(results, lables={"cmod_tuples": [], "marker": []}):
    for result in results:
        if (result["c"], result["mod"]) in lables["cmod_tuples"]:
            result["lable"] = lables["marker"][
                lables["cmod_tuples"].index((result["c"], result["mod"]))
            ]
        else:
            result["lable"] = ""
    return results


def cap2vec(capacities, maxdel=1000, maxdeg=25, threshold=0.05):
    vec = np.zeros((maxdel, maxdeg))
    for idx in range(len(capacities)):
        delay = capacities[idx]["delay"]
        degree = capacities[idx]["degree"]
        if (delay <= maxdel) and (degree <= maxdeg):
            if capacities[idx]["score"] > threshold:
                vec[delay - 1, degree - 1] += capacities[idx]["score"]
    return vec


def capacity_lables(folder, data_type="p"):
    cap_data = load_results(folder, data_type="p")
    labels = {}
    labels["cmod_tuples"] = []
    labels["marker"] = []
    for data in cap_data:
        vec = cap2vec(data["allcaps"])
        data["totalcap"] = np.sum(vec)
        labels["cmod_tuples"].append((data["c"], data["mod"]))
        labels["marker"].append(f"{np.round(data['totalcap'],2)}")
    return labels


def pretty_heatmap(
    folder,
    ax,
    name,
    cmap,
    y,
    x,
    title="",
    fontsize=18,
    cmap_lim=[None, None],
    cbar_shrink=0.7,
):
    sns.set(font_scale=1.4)
    results = load_results(folder, trans_cap_data=True)
    results = pd.DataFrame(results)
    value = results.pivot(y, x, name)
    sns.heatmap(
        value,
        ax=ax,
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": cbar_shrink},
        vmin=cmap_lim[0],
        vmax=cmap_lim[1],
    )
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)))
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontsize=fontsize)
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(labelsize=16)
    ax.text(0, -0.21, title, fontsize=fontsize)
    return ax


def plot_cmod_heatmap(folder, folder_caps=False, lable=False, plotcap=True):
    results = load_results(folder, trans_cap_data=True)
    if not lable:
        results = set_lables(results)
        results = pd.DataFrame(results)
    else:
        results = set_lables_cmod(results, lable)
        results = pd.DataFrame(results)
    if folder_caps:
        cap_res = load_results(folder_caps, data_type="p", trans_cap_data=True)
        for cap in cap_res:
            cap["o1"] = np.round(np.sum(cap["vec"][:, 0]), 2)
            cap["o2"] = np.round(np.sum(cap["vec"][:, 1]), 2)
            cap["o3"] = np.round(np.sum(cap["vec"][:, 2]), 2)
            cap["total1d"] = np.sum(cap["vec"][0, :])
            cap["total2d"] = np.sum(cap["vec"][1, :])
            cap["total3d"] = np.sum(cap["vec"][2, :])
        cap_res = pd.DataFrame(cap_res)
    spectral = sns.color_palette("Spectral", as_cmap=True)
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    coolwarm_r = sns.color_palette("coolwarm_r", as_cmap=True)
    lables = results.pivot("c", "mod", "lable").values

    if not plotcap:
        fig, axs = plt.subplots(2, 3)
    else:
        fig, axs = plt.subplots(3, 3)

    # average rate
    avg_rate = results.pivot("c", "mod", "avg_rate")
    sns.heatmap(
        avg_rate, ax=axs[0][0], cmap=coolwarm, vmin=0.0, annot=lables, fmt=""
    )  # , vmax=50.
    axs[0][0].set_title("average rate")

    # average rate
    avg_rate = results.pivot("c", "mod", "avg_nmda_length")
    sns.heatmap(avg_rate, ax=axs[0][1], cmap=coolwarm, annot=lables, fmt="")
    axs[0][1].set_title("average nmda length")

    # average rate
    avg_rate = results.pivot("c", "mod", "relative_upstate")
    sns.heatmap(avg_rate, ax=axs[0][2], cmap=coolwarm, annot=lables, fmt="")
    axs[0][2].set_title("relative upstate")

    if plotcap:
        results = cap_res
        for i in range(3):
            # plot order i+1 total cap
            avg_rate = results.pivot("c", "mod", f"o{i+1}")
            sns.heatmap(avg_rate, ax=axs[1][i], cmap=coolwarm_r)
            axs[1][i].set_title(f"summed capacity of order {i+1}")
            # plot total cap for delay i+1
            avg_rate = results.pivot("c", "mod", f"total{i+1}d")
            sns.heatmap(avg_rate, ax=axs[2][i], cmap=coolwarm_r)
            axs[2][i].set_title(f"summed capacity of delay {i+1}")
    plt.tight_layout()


def plot_rg_heatmaps(folder, lable=False):
    results = load_results(folder)
    if not lable:
        results = set_lables(results)
    else:
        results = set_lables(results, lable)
        results = pd.DataFrame(results)
    spectral = sns.color_palette("Spectral", as_cmap=True)
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    coolwarm_r = sns.color_palette("coolwarm_r", as_cmap=True)
    # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    fig, axs = plt.subplots(3, 3, constrained_layout=True)

    lables = results.pivot("r_bg", "g", "lable").values

    # average cc
    avg_cc = results.pivot("r_bg", "g", "avg_cc")

    sns.heatmap(
        avg_cc,
        ax=axs[0][0],
        cmap=coolwarm,
        vmin=0.0,
        vmax=0.25,
        annot=lables,
        fmt="",
    )
    axs[0][0].set_title("cross correlation")

    # average cv
    avg_cv = results.pivot("r_bg", "g", "avg_cv")
    sns.heatmap(
        avg_cv,
        ax=axs[0][1],
        cmap=coolwarm_r,
        vmin=0.0,
        vmax=1.0,
        annot=lables,
        fmt="",
    )
    axs[0][1].set_title("coefficient of variation")
    axs[0][1].set(yticklabels=[])
    axs[0][1].set(ylabel="")

    # average pu20
    pu20 = results.pivot("r_bg", "g", "pu20")
    sns.heatmap(
        pu20,
        ax=axs[0][2],
        cmap=coolwarm_r,
        vmin=0.0,
        vmax=100.0,
        annot=lables,
        fmt="",
    )
    axs[0][2].set_title("% " + " with r < 20Hz")
    axs[0][2].set(yticklabels=[])
    axs[0][2].set(ylabel="")

    # average rate
    avg_rate = results.pivot("r_bg", "g", "avg_rate")
    sns.heatmap(
        avg_rate,
        ax=axs[1][0],
        cmap=coolwarm,
        vmin=0.0,
        vmax=50.0,
        annot=lables,
        fmt="",
    )
    axs[1][0].set_title("average rate")

    # avg_ex_rate
    avg_ex_rate = results.pivot("r_bg", "g", "avg_ex_rate")
    sns.heatmap(
        avg_ex_rate,
        ax=axs[1][1],
        cmap=coolwarm,
        vmin=0.0,
        vmax=50.0,
        annot=lables,
        fmt="",
    )
    axs[1][1].set_title("avg excitatory rate")
    axs[1][1].set(yticklabels=[])
    axs[1][1].set(ylabel="")

    # avg_inh_rate
    avg_inh_rate = results.pivot("r_bg", "g", "avg_inh_rate")
    sns.heatmap(
        avg_inh_rate,
        ax=axs[1][2],
        cmap=coolwarm,
        vmin=0.0,
        vmax=50.0,
        annot=lables,
        fmt="",
    )
    axs[1][2].set_title("avg inhibitory rate")
    axs[1][2].set(yticklabels=[])
    axs[1][2].set(ylabel="")

    # average max avg rate
    max_avg_rate = results.pivot("r_bg", "g", "max_avg_rate")
    sns.heatmap(max_avg_rate, ax=axs[2][0], cmap=coolwarm, annot=lables, fmt="")
    axs[2][0].set_title("max_avg sg neuron rate")

    # average max N silent
    # n_silent = results.pivot('r_bg', 'g', 'N_silent')
    # sns.heatmap(n_silent, ax=axs[1][1], cmap=coolwarm)
    # axs[1][1].set_title('n silent neurons')
    # axs[1][1].set(yticklabels=[])
    # axs[1][1].set(ylabel='')

    # average NMDA length
    nmda_length = results.pivot("r_bg", "g", "avg_nmda_length")
    sns.heatmap(nmda_length, ax=axs[2][1], annot=lables, fmt="")
    axs[2][1].set_title("avg NMDA length")
    axs[2][1].set(yticklabels=[])
    axs[2][1].set(ylabel="")

    # relatve upstate
    relative_upstate = results.pivot("r_bg", "g", "relative_upstate")
    sns.heatmap(relative_upstate, ax=axs[2][2], annot=lables, fmt="")
    axs[2][2].set_title("relative upstate")
    axs[2][2].set(yticklabels=[])
    axs[2][2].set(ylabel="")

    return fig, axs


def plot_cap_histogram(path, threshold):
    with open(path, "rb") as file:
        result = pickle.load(file)
        caplist = [
            cap["score"] #(cap["delay"], )
            for cap in result["allcaps"]
            if cap["score"] > threshold
        ]
    #breakpoint()
    plt.hist(caplist, density=False, bins=100)
    plt.show()


def plot_capacity_grid(folder, x="mod", y="c"):
    results = load_results(folder, trans_cap_data=True, data_type="p")
    x_list = np.unique([result[x] for result in results])
    y_list = np.unique([result[y] for result in results])
    plt.subplots_adjust(bottom=0.4, right=1.04, wspace=0.3, hspace=0.4)
    fig, axs = plt.subplots(len(y_list), len(x_list), sharey=False)
    i = 0
    for xidx, x_val in enumerate(x_list):
        for yidx, y_val in enumerate(y_list):
            for result in results:
                if x_list[xidx] == result[x] and y_list[yidx] == result[y]:
                    i += 1
                    # axs[yidx][xidx].set_title(x + f'= {x_val}' + y + f'= {y_val}', size = 17.)
                    plot_cap_bars(result["vec"], axs=axs[yidx][xidx])
                    break
    axs[3][3].legend(bbox_to_anchor=(-1.45, -0.25), loc="upper center", ncol=10)

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][3].set_xticklabels([])
    axs[1][0].set_xticklabels([])
    axs[1][1].set_xticklabels([])
    axs[1][2].set_xticklabels([])
    axs[1][3].set_xticklabels([])
    axs[2][0].set_xticklabels([])
    axs[2][1].set_xticklabels([])
    axs[2][2].set_xticklabels([])
    axs[2][3].set_xticklabels([])

    axs[0][1].set_yticklabels([])
    axs[0][2].set_yticklabels([])
    axs[0][3].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])
    axs[1][3].set_yticklabels([])
    axs[2][1].set_yticklabels([])
    axs[2][2].set_yticklabels([])
    axs[2][3].set_yticklabels([])
    axs[3][1].set_yticklabels([])
    axs[3][2].set_yticklabels([])
    axs[3][3].set_yticklabels([])

    axs[0][0].set_ylabel("total capacity")
    axs[1][0].set_ylabel("total capacity")
    axs[2][0].set_ylabel("total capacity")
    axs[3][0].set_ylabel("total capacity")
    axs[3][0].set_xlabel("maximal delay")
    axs[3][1].set_xlabel("maximal delay")
    axs[3][2].set_xlabel("maximal delay")
    axs[3][3].set_xlabel("maximal delay")

    axs[0][0].text(
        -0.25,
        0.5,
        "c=0.2",
        transform=axs[0][0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[1][0].text(
        -0.25,
        0.5,
        "c=0.4",
        transform=axs[1][0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[2][0].text(
        -0.25,
        0.5,
        "c=0.6",
        transform=axs[2][0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[3][0].text(
        -0.25,
        0.5,
        "c=0.8",
        transform=axs[3][0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )

    axs[3][0].text(
        0.65,
        1.15,
        "m=0.0",
        transform=axs[0][0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[3][1].text(
        0.65,
        1.15,
        "c=0.3",
        transform=axs[0][1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[3][2].text(
        0.65,
        1.15,
        "c=0.6",
        transform=axs[0][2].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axs[3][3].text(
        0.65,
        1.15,
        "c=0.9",
        transform=axs[0][3].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )

    return fig, axs


def plot_capacities(pathorvec, axs=None):

    # load pickles with capacity results
    maxdel = 8
    maxdeg = 20
    delrange = np.arange(1, maxdel)
    if axs == None:
        with open(pathorvec, "rb") as f:
            results = pickle.load(f)
        totalcap, allcaps, numcaps, nodes = (
            results["totalcap"],
            results["allcaps"],
            results["numcaps"],
            results["nodes"],
        )
        vec = cap2vec(allcaps, maxdel=maxdel, maxdeg=maxdeg)
        fig, axs = plt.subplots(1)
    else:
        vec = pathorvec
    totalcap = np.sum(vec, axis=1)
    axs.semilogy(delrange, vec[:, 0][delrange], label="linear")
    axs.set_ylim(10**-3, 10**1)
    axs.grid()
    axs.semilogy(delrange, vec[:, 1][delrange], label="qadratic")
    axs.semilogy(delrange, vec[:, 2][delrange], label="qubic")
    axs.semilogy(delrange, vec[:, 3][delrange], label="$\mathcal{o}^4$")
    axs.set_xlabel("maximal delay")
    axs.set_ylabel("total capacity")
    plt.show()
    return axs


def plot_cap_bars(pathorvec, maxdeg=10, axs=None):

    # load pickles with capacity results
    maxdel = 8
    delrange = np.arange(1, maxdel)
    if axs == None:
        with open(pathorvec, "rb") as f:
            results = pickle.load(f)
        totalcap, allcaps, numcaps, nodes = (
            results["totalcap"],
            results["allcaps"],
            results["numcaps"],
            results["nodes"],
        )
        vec = cap2vec(allcaps, maxdel=maxdel, maxdeg=maxdeg)
        fig, axs = plt.subplots(1)
    else:
        vec = pathorvec
    # axs.set_yscale('log')
    axs.set_ylim(0, 8.5)
    axs.set_xlim(0.5, 5.5)
    bottom = np.zeros(len(delrange))
    for i in range(len(pathorvec[0]) - 1):
        values = vec[:, i][delrange]
        axs.bar(delrange, values, label=f"O({i+1})", bottom=bottom, lw=0.0)
        bottom += values
    return axs


def pretty_cap_bar(res_path, ax, title, degree, maxdeg=8):
    # load pickles with capacity results
    maxdel = 8
    maxdeg = maxdeg
    delrange = np.arange(1, maxdel)
    with open(res_path, "rb") as f:
        results = pickle.load(f)
    totalcap, allcaps, numcaps, nodes = (
        results["totalcap"],
        results["allcaps"],
        results["numcaps"],
        results["nodes"],
    )
    vec = cap2vec(allcaps)
    pltst.myAx(ax)
    ax.tick_params(labelsize=16)
    ax.text(2, 0.75, title, fontsize=14)
    ax.set_ylabel("memory", fontsize=19)
    ax.set_xlabel("time steps", fontsize=19)
    ax.set_xlim(0.5, 6.5)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.xaxis.set_ticklabels([1, 2, 3, 4, 5, 6])
    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**0)
    ax = pltst.myAx(ax)
    ax.set_yticks([10**-2, 10**-1, 10**0])
    if degree == "all":
        values = np.sum(vec, axis=1)[delrange]
        ax.set_ylim(10**-3, 5 * 10**0)
        ax.bar(delrange, values)
        return ax
    else:
        values = vec[:, degree][delrange]
        ax.bar(delrange, values)
        return ax


def pretty_allcap(res_path, ax, title, maxdeg=8):
    # load pickles with capacity results
    maxdel = 8
    maxdeg = maxdeg
    delrange = np.arange(1, maxdel)
    with open(res_path, "rb") as f:
        results = pickle.load(f)
    totalcap, allcaps, numcaps, nodes = (
        results["totalcap"],
        results["allcaps"],
        results["numcaps"],
        results["nodes"],
    )
    vec = cap2vec(allcaps)
    pltst.myAx(ax)
    ax.tick_params(labelsize=16)
    ax.text(2, 4.5, title, fontsize=14)
    ax.set_ylabel("memory", fontsize=19)
    ax.set_xlabel("time steps", fontsize=19)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, 5.5)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.xaxis.set_ticklabels([1, 2, 3, 4, 5, 6])
    ax = pltst.myAx(ax)
    bottom = np.zeros(len(delrange))
    for degree in range(maxdeg):
        values = vec[:, degree][delrange]
        ax.bar(delrange, values, bottom=bottom, label=f"{degree}")
        bottom += values
    return ax


def plot_transition_stats(
    folder,
    axs=None,
    plotlist=[
        "r_bg",
        "avg_cc",
        "avg_cv",
        "avg_rate",
        "relative_upstate",
        "avg_nmda_length",
        "N_silent",
    ],
    cidx=0,
):
    # plot plotlist[1:] as a function of plotlist[0]
    if not isinstance(axs, np.ndarray):
        fig, axs = plt.subplots(len(plotlist) - 1)
    else:
        axs = axs[0 : len(plotlist) - 1]
    styles = ["-", "-", "y^-"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    results = pd.DataFrame(load_results(folder))
    results = results[plotlist]
    results = results.sort_values(by="r_bg")
    results.set_index("r_bg").plot(ax=axs, subplots=True, color=colors[cidx])
    return axs


def plot_scaling(
    folder, axs=None, color="black", plotlist=["cores", "time"], cidx=0
):
    results = pd.DataFrame(load_results(folder))
    results = results[plotlist]
    results = results.sort_values(by="cores")
    results.set_index("cores").plot(ax=axs, subplots=True, color=color)
    return axs


def plot_capacity_transition(
    folder, axs, plotlist=["r_bg", "totalcap", "lin_cap"]
):
    results = load_results(folder, trans_cap_data=True, data_type="p")
    for result in results:
        result["lin_cap"] = np.sum(result["vec"][:, 0])
        vec = cap2vec(result["allcaps"])
        result["totalcap"] = np.sum(vec)
    results = pd.DataFrame(results)[plotlist]
    results = results.sort_values(by="r_bg")
    results.set_index("r_bg").plot(ax=axs, subplots=True)


def plot_population_rates(raw_data, axs, results, dt=5.0):
    # plot poplation firing rates for excitatory inhibitory and

    spikelist = sg.create_SpikeList(
        raw_data, N=results["N"], t_start=2000, t_stop=2000 + raw_data["t_sim"]
    )
    ex_spikelist = spikelist.id_slice(raw_data["excitatory_ids"])
    inh_spikelist = spikelist.id_slice(raw_data["inhibitory_ids"])
    spikelistlist = [spikelist, ex_spikelist, inh_spikelist]
    label = ["full", "excitatory", "inhibitory"]
    colors = ["tab:blue", "tab:green", "tab:olive"]
    for i, splist in enumerate(spikelistlist):
        time = splist.time_axis(dt)[:-1]
        rate = splist.firing_rate(dt, average=True)
        axs.plot(time, rate, label=label[i], color=colors[i])
    axs.legend()


def rasta_nmda_only(
    ax,
    raw_data,
    results,
    plotrange=(2000, 3000),
    l_min=30,
    N=150,
    plot_seq=False,
):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    with open(results, "rb") as file:
        results = json.load(file)

    t_start = 2000
    t_stop = int(2000 + 1100)
    timecut = np.where(data["sample_neurons"]["times"] < 3100)
    data["sample_neurons"]["times"] = data["sample_neurons"]["times"][timecut]
    data["sample_neurons"]["senders"] = data["sample_neurons"]["senders"][
        timecut
    ]

    # basic rasta plot with nmda spikes as red lines
    nmda_data = sg.find_NMDA(data, -35.0, 30, "sample_neurons")

    # ploting nmdas
    total_active = np.zeros(t_stop - t_start)
    leaf_idxs = data["sample_neurons"]["leaf_idxs"]
    for neuron_id in nmda_data:
        for leaf_idx in leaf_idxs:
            for nmda in nmda_data[neuron_id][f"nmda_comp{leaf_idx}"][
                "nmda_pos"
            ]:
                if nmda[0] < 3000.0:
                    # length of nmda
                    time_array = np.arange(nmda[0], nmda[1], dtype=int)
                    total_active[time_array] += 1  # calc total active nmdas
                    offset = np.ones(len(time_array), dtype=int) * 2000
                    time_array = time_array + offset
                    id_array = np.ones(len(time_array), dtype=int) * int(
                        neuron_id
                    )
                    ax.plot(time_array, id_array, "b", alpha=0.02)
    # plotting the spikelist with prerun to optionally show it
    spikelist = sg.create_SpikeList(
        data, N=results["N"], t_start=2000, t_stop=3000
    )
    spikelist = spikelist.id_slice(data["excitatory_ids"])
    spikelist.raster_plot(
        with_rate=False,
        ax=ax,
        N=N,
        display=False,
        save=False,
        color=colours[4],
        marker="|",
        markersize=1.6,
    )

    ax.set_xlim(plotrange)
    # ax = pltst.drawScaleBars(ax, xlabel='ms', ylabel = 'IDs')

    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)

    return ax


def rasta_nmda(
    raw_data, results, plotrange=(2000, 3000), l_min=30, N=250, plot_seq=False, ax=None, partial_return=False
):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
        #breakpoint()
    with open(results, "rb") as file:
        results = json.load(file)

    t_start = 2000
    t_stop = int(2000 + data["t_sim"])

    # basic rasta plot with nmda spikes as red lines
    nmda_data = sg.find_NMDA(data, -35.0, 30)
    avg_rel_up = np.sum(
        [nmda_data[neuron_id]["rel_up"] for neuron_id in nmda_data]
    ) / len(nmda_data)
    fig, axs = plt.subplots(3)
    
    if ax != None:
        axs[0] = ax
    
    # axs[0].fill_between(np.arange(t_start, t_stop), np.zeros(t_stop-t_start), np.ones(t_stop-t_start)*int(results['N']*0.8), facecolor='blue', alpha=0.2, label = 'excitatory')
    axs[0].fill_between(
        np.arange(t_start, t_stop),
        np.ones(t_stop - t_start) * int(results["N"] * 0.8),
        np.ones(t_stop - t_start) * results["N"],
        facecolor="orange",
        alpha=0.2,
        label="inhibitory",
    )
    
    cluster = int(results["N_cl"])
    ncol = 1
    fontsize = 9
    cmap = plt.cm.get_cmap('viridis')
    
    if cluster > 20:
        ncol = 3
        fontsize = 8
    elif cluster > 10:
        ncol = 2
    
    #identify clusters by colorblocking
    for i in range(0, cluster):
        
        if i%2 == 0:
            color = cmap(1/cluster * i/2)
        else:
            color = cmap(1/cluster * (cluster - (i - 1)/2))
        
        axs[0].fill_between(
            np.arange(t_start, t_stop),
            np.ones(t_stop - t_start) * int(results["N"] * 0.8 * 1/cluster * (cluster - i)),
            np.ones(t_stop - t_start) * int(results["N"] * 0.8 * 1/cluster * (cluster - (i + 1))),
            facecolor=color,
            alpha=0.2,
            label="cluster {}".format(i + 1),
        )

    if hasattr(data["seq"], "__len__"):
        ax_seq = axs[0].twinx()
        ax_seq.plot(
            np.arange(len(data["seq"])) * data["tstep"]
            + np.ones(len(data["seq"])) * 2000,
            data["seq"],
            color="green",
        )
        ax_seq.set_ylim(-1.25, 1.875)

    # plotting the spikelist with prerun to optionally show it
    spikelist = sg.create_SpikeList(
        data, N=results["N"], t_start=0, t_stop=t_stop
    )
    spikelist.raster_plot(
        with_rate=False, ax=axs[0], N=N, dt=5, display=False, save=False
    )

    # plot population rates
    plot_population_rates(data, axs[1], results)

    # ploting nmdas
    total_active = np.zeros(t_stop - t_start)
    leaf_idxs = data["sample_neurons"]["leaf_idxs"]
    for neuron_id in nmda_data:
        for leaf_idx in leaf_idxs:
            for nmda in nmda_data[neuron_id][f"nmda_comp{leaf_idx}"][
                "nmda_pos"
            ]:
                # length of nmda
                time_array = np.arange(nmda[0], nmda[1], dtype=int)
                total_active[time_array] += 1  # calc total active nmdas
                offset = np.ones(len(time_array), dtype=int) * 2000
                time_array = time_array + offset
                id_array = np.ones(len(time_array), dtype=int) * int(neuron_id)
                axs[0].plot(time_array, id_array, "r", alpha=0.05)

    ax_upstate = axs[1].twinx()
    ax_upstate.set_ylabel("$N_{NMDA}$", color="red")
    ax_upstate.plot(np.arange(t_start, t_stop), total_active, color="red")
    ax_upstate.set_xlim(plotrange[0], plotrange[1])
    axs[0].set_xlim(plotrange[0], plotrange[1])
    axs[0].set_title(
        "average relative time in upstate is {} %".format(avg_rel_up)
    )
    axs[1].set_xlim(plotrange[0], plotrange[1])
    axs[0].legend(bbox_to_anchor=(1.001, 1), loc="upper left", fontsize=fontsize, ncol=ncol)
    
    #breakpoint()
    if partial_return:
        return axs[0]

    # firing rate histogram --> new spikelist without prerun
    spikelist = sg.create_SpikeList(
        data, N=results["N"], t_start=t_start, t_stop=t_start + data["t_sim"]
    )  # 1000. should be t_sim
    mean_rates = spikelist.mean_rates()
    sns.histplot(mean_rates, ax=axs[2], binwidth=0.5, binrange=(0.0, 30.0))
    fig.suptitle(
        "mod = {}, c = {}, g = {}, r = {} \n N_s = {}, cc = {}, cv ={}, inp_str={}".format(
            results["mod"],
            results["c"],
            results["g"],
            results["r_bg"],
            results["N_silent"],
            results["avg_cc"],
            results["avg_cv"],
            results["inp_str"],
        )
    )

    #plt.tight_layout()
    
    return fig, axs


def pretty_nmda_traces(
    ax,
    raw_data,
    neuron_id,
    color,
    comps=["v_comp0", "v_comp2", "v_comp5"],
    time_int=(2000, 3000),
):
    time_int = (time_int[0] * 10, time_int[1] * 10)
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    data = sg.find_NMDA(data, -35.0, 30, name="plottrace")

    neuron_ids = list(data.keys())
    idd = neuron_ids[neuron_id]
    print(idd)
    neuron_id = neuron_ids[neuron_id]
    time = np.arange(time_int[0], time_int[1]) * 0.1
    for idx, comp in enumerate(comps):
        ax.plot(
            time,
            data[neuron_id][comp][time_int[0] : time_int[1]],
            label=comp,
            color=color[idx],
            linewidth=0.8,
        )
    ax.set_ylim((-90.0, 40.0))
    ax.set_xlim(time[0], time[-1])
    # ax1 = ax.twiny()
    # ax1.set_xlim(time_int)
    # ax1.set_xticks([2000,2100])
    ax.set_title("neuron id: {}".format(idd))
    ax.patch.set_alpha(0.01)
    ax = pltst.drawScaleBars(ax, xlabel="ms", ylabel="mV")
    
    return ax


def nmda_traces(raw_data, N, time_int=(4000, 5000), ordered=True):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    v_comps = ["v_comp0"] + data["sample_neurons"]["v_comps"]
    data = sg.find_NMDA(data, -35.0, 30)
    nmda_data = pd.DataFrame([data[neuron] for neuron in data])
    if ordered:
        samples_ids = (
            nmda_data.sort_values(by=["rel_up"], ascending=False)
            .head(N)
            .pop("neuron_id")
            .values.tolist()
        )
    else:
        samples_ids = np.random.choice(range(1, len(data)), N)
    fig, axs = plt.subplots(N)
    data = [data[f"{samples_id}"] for samples_id in samples_ids]
    time = np.arange(time_int[0], time_int[1])

    for i in range(N):
        for comp in v_comps:
            axs[i].plot(
                time, data[i][comp][time_int[0] : time_int[1]], label=comp
            )
    plt.legend()
    return fig, axs


def simple_rasta(ax, data, N=250, with_rate=False):
    fig, axs = plt.subplots(2)
    spikelist = sg.create_SpikeList(data, N=1250, t_start=2000, t_stop=22000)
    spikelist.raster_plot(
        with_rate=with_rate, ax=ax, N=N, dt=5, display=False, save=False
    )
    return ax


def inp_rasta(ax, raw_data, plot_range, perc=0.05):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    if hasattr(data["seq"], "__len__"):
        ax_seq = ax.twinx()
        ax_seq.plot(
            np.arange(1, len(data["seq"]) + 1) * data["tstep"]
            + np.ones(len(data["seq"])) * 2000,
            data["seq"],
            color="pink",
            drawstyle="steps",
        )
        ax_sep = pltst.noFrameAx(ax_seq)
        ax_seq.set_ylim(-1.25, 1.25)
    data = data["inp"]
    timecut = np.where(data["times"] < 3100)[0]
    data["times"] = data["times"][timecut]
    data["senders"] = data["senders"][timecut]
    ids = np.random.choice(
        len(data["senders"]), int(perc * len(data["senders"]))
    )
    ax.plot(
        data["times"][ids],
        data["senders"][ids],
        "|",
        markersize=1.6,
        color="black",
        alpha=0.5,
    )
    ax.set_xlim(plot_range)
    ax.set_ylim(np.min(data["senders"]), np.max(data["senders"]))
    ax = pltst.noFrameAx(ax)
    return ax


def cap_test_plt(raw_data, plotrange=(1900, 4000)):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    fig, axs = simple_rasta(data)
    axs[0].set_xlim(plotrange)
    ax_seq = axs[0].twinx()
    ax_seq.plot(
        np.arange(len(data["seq"])) * data["t_step"]
        + np.ones(len(data["seq"])) * 2000,
        data["seq"],
        color="pink",
    )
    ax_seq.set_ylim(-1.25, 1.875)

    return fig, axs


def plot_input(ax, N, inp_str, rate, value):
    stim_id = np.linspace(int(N / 100), N, int(N / 10))
    zero_point = int(N / 2)
    std = float(N / 20)
    interval = 0.4 * N

    # both stim_rates and base rates vec are normalized to N
    # stim_rates = np.array([stats.norm.pdf(stim_id, ccro_point+ interval*val, std)*N for val in inp_seq])
    stim_rates = stats.norm.pdf(stim_id, zero_point + interval * value, std) * N
    base_rates = np.ones(int(N / 10))

    rate_values = (
        ((1 - inp_str) * base_rates + inp_str * stim_rates) * rate * 0.1 * N
    )
    # rate_values = ((base_rates + inp_str*stim_rates) * inp_rate * 0.1 * N).tolist()
    ax.bar(
        stim_id,
        base_rates * (1 - inp_str) * rate,
        width=-10.0,
        align="edge",
        label="background",
    )
    ax.bar(
        stim_id,
        stim_rates * inp_str * rate,
        bottom=base_rates * (1 - inp_str) * rate,
        width=-10.0,
        align="edge",
        label="signal",
    )
    return ax


def plot_inputdemo(values, N=1000, inp_str=0.1, rate=12):
    fig, axs = plt.subplots(len(values))
    ax_inp = [None] * len(values)
    for i, ax in enumerate(axs):
        plot_input(axs[i], N, inp_str, rate, values[i])
        axs[i].set_xlim(0, 1000)
        axs[i].set_ylabel("rate")
        axs[i].set_xlim(0, 1000)
        ax_inp[i] = axs[i].twiny()
        ax_inp[i].plot([values[i], values[i]], [0, 22], "black")
        ax_inp[i].set_xlim(-1.25, 1.25)
    axs[0].legend()
    ax_inp[0].set_xlabel("random input")
    axs[-1].set_xlabel("excitatory neurons")
    return fig, axs


def plot_dendhists(raw_data, id_list):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    N_dend = data["N_dend"]
    fig, axs = plt.subplots(len(id_list))
    for i, neuron_id in enumerate(id_list):
        dendhist(data["hists"][f"{neuron_id}"], N_dend, axs[i])
    return fig, axs


def single_dendhist(ax, raw_data, neuron_id, color, N_dend=5):
    with open(raw_data, "rb") as file:
        data = pickle.load(file)
    ax = dendhist(data["hists"][f"{neuron_id}"], N_dend, ax, color=color)
    return ax


def dendhist(hist, N_dend, ax, color):
    bottom = np.zeros(N_dend)
    for cluster in range(N_dend):
        keys = np.array(list(hist.keys())[1:])
        values = np.array([hist[key][str(cluster)] for key in keys])
        ax.bar(
            [1, 2, 3, 4, 5],
            values,
            label=cluster,
            bottom=bottom,
            color=color[cluster],
        )
        bottom += values
    return ax


def plot_modularity(ax, mod, n_cl, n_e, color):
    def k_intra(n_cl, n_e, m):
        # return 0.1*n_e/(1 + (n_cl - 1)*(1-m))
        return 1 / (1 + (n_cl - 1) * (1 - m))

    def k_inter(n_cl, n_e, m):
        # return 0.1*n_e*(1-m)/(1+(n_cl -1)*(1-m))
        return 1 * (1 - m) / (1 + (n_cl - 1) * (1 - m))

    cluster_ids = np.arange(1, n_cl + 1)
    connections = {cl_id: k_inter(n_cl, n_e, mod) for cl_id in cluster_ids}
    connections[1] = k_intra(n_cl, n_e, mod)
    for idx, cluster in enumerate(cluster_ids):
        ax.bar(cluster_ids[idx], connections[cluster], color=color[idx])
    ax = pltst.myAx(ax)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim((0, 1))
    return ax


def plot_cap_heatmap(folder_caps=False, lable=False, plotcap=True):
    cap_res = load_results(folder_caps, data_type="p", trans_cap_data=True)
    for cap in cap_res:
        cap["o1"] = np.round(np.sum(cap["vec"][:, 0]), 2)
        cap["o2"] = np.round(np.sum(cap["vec"][:, 1]), 2)
        cap["o3"] = np.round(np.sum(cap["vec"][:, 2]), 2)
        cap["total1d"] = np.sum(cap["vec"][0, :])
        cap["total2d"] = np.sum(cap["vec"][1, :])
        cap["total3d"] = np.sum(cap["vec"][2, :])
    cap_res = pd.DataFrame(cap_res)
    spectral = sns.color_palette("Spectral", as_cmap=True)
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    coolwarm_r = sns.color_palette("coolwarm_r", as_cmap=True)
    # lables = results.pivot('c', 'mod', 'lable').values

    fig, axs = plt.subplots(1, 2)

    allcaps = cap_res.pivot("c", "mod", "totalcap")
    sns.heatmap(allcaps, ax=axs[0], cmap=coolwarm_r)

    current = cap_res.pivot("c", "mod", "total1d")
    sns.heatmap(current, ax=axs[1], cmap=coolwarm_r)
    """
    if plotcap:
        results = cap_res
        for i in range(3):
            #plot order i+1 total cap 
            avg_rate = results.pivot('c', 'mod', f'o{i+1}')
            
            axs[1][i].set_title(f'summed capacity of order {i+1}')
            #plot total cap for delay i+1     
            avg_rate = results.pivot('c', 'mod', f'total{i+1}d')
            sns.heatmap(avg_rate, ax=axs[2][i], cmap=coolwarm_r)
            axs[2][i].set_title(f'summed capacity of delay {i+1}')
    """
    plt.tight_layout()
    plt.show()
    
    
def plot_transition_rate_graph(
    res,
    ax,
    plot_name,
    lable_name,
    color,
    run=0,
    title="transition_rate_test",
    fontsize=18,
):
    #results = load_results(folder)
    label = "mod: {}, c: {}".format(res[0]["mod"], res[0]["c"])
    res = pd.DataFrame(res)
    res.sort_values(by=['r_bg'], inplace=True)
    rate = res.loc[:,"r_bg"]
    value = res.loc[:,plot_name]
    ax.plot(rate, value, color=color, label=label) #'tab:orange'
    #cmap = plt.cm.get_cmap('magma')
    #color = cmap(0.9) light red
    ax.set_facecolor('0.8')
    ax.patch.set_alpha(0.2)
    ax.grid(axis='y')
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    #ax.set_yticks(np.arange(ylow, yhigh, yticks))
    y_label = lable_name
    ax.set_ylabel(y_label, fontsize=fontsize)
    if run == 0:
        ax.legend(bbox_to_anchor=(1.001, 1), loc="upper left", fontsize=10)
    if run == 3:
        x_label = "background rate [Hz]"
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(labelsize=16)
        ax.text(0, -0.21, title, fontsize=fontsize)
        
    return ax
    
    
def look_up(value):
    if value == 0.0:
        return 0
    elif value == 0.2:
        return 1
    elif value == 0.4:
        return 2
    elif value == 0.6:
        return 3
    elif value == 0.8:
        return 4
    elif value == 0.9:
        return 5
        
def look_up_threshold(n):
    #breakpoint()
    if n == 2500:
        return 0.055452/2
    elif n == 5000:
        return 0.10763/2
        
def look_up_r(n, mod, c):
    rates = [10.65, 10.65, 10.6, 10.55, 10.65, 10.6, 10.6, 10.6, 10.65, 10.65, 10.6, 10.6, 10.6, 10.6, 10.6, 10.6, 10.5, 10.6, 10.65, 10.7, 10.7, 10.7, 10.7, 10.7,
            #n2500
            7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.6, 7.55, 7.55, 7.6, 7.6,
            #n5000
            5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45, 5.45]
            #n10000
        
    i = int(round(n/2500, 1)) #1,2,3
    j = int(round(mod/0.2, 1)) #0,1,2,3,4, 5
    if mod == 0.9:
        j = 5
    k = int(round(c/0.2, 1)) # 1,2,3,4
    #breakpoint()
    return rates[(i - 1) * 24 + j * 4 + (k - 1)]


if __name__ == "__main__":
    
    #############################################
    ##          rasta nmda plotting            ##
    #############################################
    
    if argv [1] == "rasta_nmda":
        # fig, axs = plt.subplots(2)
        path = "/home/noah/Thesis_Git/clustered_memory_network/output/" + argv[2]
        path_raw = "raw/" + argv[3] #"raw/raw_r15.0_g14.0_mod0.0_c0.2_str0.1_step50.0_weight0.00025.p" 
        path_res = "res/" + argv[4] #"res/_r15.0_g14.0_mod0.0_c0.2_str0.1_step50.0_weight0.00025.json"
        rasta_nmda(
            path
            + path_raw,
            path
            + path_res,
            plotrange=(0000, 4000),
        )
        plt.show()
    
    #############################################
    ## custom rasta nmda plus traces  plotting ##
    #############################################
    
    elif argv [1] == "rasta_nmda_plus_traces":
        
        mosaic = """AA;..;.B;..;.C;..;.D;..;.E;..;.F""" #.G;.H;.I;.J;.K"""
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(mosaic, height_ratios=[5,0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75,0.25,1.75], width_ratios=[0.975,1.025])
        #identify_axes(ax_dict)
        
        path = "/home/noah/Thesis_Git/clustered_memory_network/output/" + argv[2]
        path_raw = "raw/" + argv[3] #"raw/raw_r15.0_g14.0_mod0.0_c0.2_str0.1_step50.0_weight0.00025.p" 
        path_res = "res/" + argv[4] #"res/_r15.0_g14.0_mod0.0_c0.2_str0.1_step50.0_weight0.00025.json"
        
        ax_dict["A"]=rasta_nmda(
                        path
                        + path_raw,
                        path
                        + path_res,
                        plotrange=(0000, 4000),
                        ax=ax_dict["A"],
                        partial_return=True,
                        )
                        
        with open(path+path_res, "rb") as file:
            results = json.load(file)
            
        fig.suptitle(
            "mod = {}, c = {}, g = {}, r = {} \n N_s = {}, cc = {}, cv ={}, inp_str={}".format(
                results["mod"],
                results["c"],
                results["g"],
                results["r_bg"],
                results["N_silent"],
                results["avg_cc"],
                results["avg_cv"],
                results["inp_str"],
            )
        )
                            
        dict_array=["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        
        for i in range (0, 5):
            pretty_nmda_traces(
                ax=ax_dict[dict_array[i]],
                raw_data=path + path_raw,
                neuron_id=i * 4,
                color=['r', 'b', 'g', 'm', 'y'],
                comps=["v_comp1", "v_comp2", "v_comp3", "v_comp4", "v_comp5"],
                time_int=(2000, 4000)
            )
            
        plt.show()
            
    #############################################
    ##          nmda trace plotting            ##
    #############################################
    
    elif argv[1] == "nmda_traces":
        fig, axs = plt.subplots(20, 1)
        path = "/home/noah/Thesis_Git/clustered_memory_network/output/" + argv[2]
        path_raw = "raw/" + argv[3]
        
        for i in range (0, 20):
            pretty_nmda_traces(
                ax=axs[i],
                raw_data=path + path_raw,
                neuron_id=i,
                color=['r', 'b', 'g', 'm', 'y'],
                comps=["v_comp0", "v_comp1", "v_comp2", "v_comp4", "v_comp5"],
                time_int=(2000, 4000),
            )
        
        plt.show()
    
    #############################################
    ##           heatmap plotting              ##
    #############################################
    
    elif argv[1] == "heatmap":
    
        fig, axs = plt.subplots(2,2)
        fig.suptitle(argv[2], fontsize=20)
    
        #usage: name_of_src_folder plot_mod_c/rbg_g
        path = "/home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/"
        path_res_folder = argv[2] + "res/"
    
        coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    
        name = ["avg_rate", "avg_cv", "avg_cc", "relative_upstate"] #column name of data to plot
        title = ["average rate [Hz]", "CV", "CC", "rel. upstate [%]"]
        cmap_lim = [None,None]
    
        x = "r_bg"
        y = "g"
    
        if argv[3] == "mod_c":
            x = "mod"
            y = "c"
    
        for i in range (0,2):
            for j in range (0,2):
            
                cmap_lim = [None,None]
                
                if i == 0 and j == 0:
                    cmap_lim = [0,20]
                elif i == 0 and j == 1 : 
                    cmap_lim = [0,1]
                elif i == 1 and j == 0:
                    cmap_lim = [0, 0.01]
                    
                pretty_heatmap(
                    path 
                    + path_res_folder,
                    ax=axs[i][j],
                    name=name[i*2 + j],
                    cmap=coolwarm,
                    y=y,
                    x=x,
                    title=title[i*2 + j],
                    fontsize=18,
                    cmap_lim=cmap_lim,
                    cbar_shrink=0.7,
                )
        
        plt.show()
        
    #############################################
    ##       transition rate plotting          ##
    #############################################
    
    elif argv[1] == "transition_rate":
    
        #usage: n 
        #plotted for transition rate of n, g 20.0, mod 0.0, c 0.2
        
        cluster = 0
        if argv[2] == "2500":
            cluster = "10"
        elif argv[2] == "5000":
            cluster = "20"
        elif argv[2] == "10000":
            cluster = "40"
            
        plot_name_array=["avg_rate", "avg_cv", "avg_cc", "avg_nmda_length", "relative_upstate"]
        lable_name_array=["avg. rate", "avg. CV", "avg. CC", "avg. nmda length", "rel NMDA[%]"]
        
        fig, axs = plt.subplots(5,1)
        fig.suptitle("n" + argv[2] + "_transition_rate_scan_g20.0_mod0.0-0.9_c0.2-0.8_cl" + cluster, fontsize=20)
        
        folder = "/home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/n" + argv[2] + "_transition_rate_scans_mod0.0-0.9_c0.2-0.8_g20.0_cl" + cluster + "_std-scaled/dend20/"
        #path_res_folder = "n" + argv[2] + "_transition_rate_scan_g20.0_cl" + cluster + "std-scaled/res/"
        
        #cmap = plt.cm.get_cmap('viridis')
        #cmap_index = 0
        color_index = 0
        
        for path in os.listdir(folder):
            res_folder = os.path.join(folder, path)
            res_folder = res_folder + "/res/"
            
            res = load_results(res_folder)
            #breakpoint()
            
            color = ['r', 'b', 'g', 'y', 'k', 'm'] #cmap(cmap_index/len(os.listdir(folder)))
            
            for i in range(0,5):
                #if i == 1:
                #    ylow = 0.35
                #    yhigh = 0.6
                #    yticks = 0.05
                #elif i == 2:
                #    ylow = 0
                #    yhigh = 0.008
                #    yticks = 0.002
                    
                #if i == 1 or i == 2:
                #    plot_transition_rate_graph(
                #        res,
                #        ax=axs[i],     
                #        plot_name=plot_name_array[i],
                #        lable_name=lable_name_array[i],
                #        title= "n" + argv[1] + "_transition_rate_scan_g20.0_mod0.0_c0.2_cl" + cluster,
                #        ylow=ylow,
                #        yhigh=yhigh,
                #        yticks=yticks,
                #    )
                #else:
                plot_transition_rate_graph(
                    res,
                    ax=axs[i],     
                    plot_name=plot_name_array[i],
                    lable_name=lable_name_array[i],
                    color=color[color_index%6],
                    run=i,
                    title= "n" + argv[1] + "_transition_rate_scan_g20.0_mod0.0_c0.2_cl" + cluster,
                )
                
            color_index += 1
        
        plt.show()
       
    #############################################
    ##         capacity test plotting          ##
    #############################################
    
    elif argv[1] == "cap_bars_test":

        # load pickles with capacity results
        results = load_results(argv[2], data_type="p")
        maxdel = 10
        maxdeg = 25
        delrange = np.arange(1, maxdel)
        fig, axs = plt.subplots(4,6)
        #loop_it = 0
        #with open(argv[2], "rb") as f:
        #    results = pickle.load(f)
        #breakpoint()
        for res in results:
            #print("new loop iteration")
            #breakpoint()
            totalcap, allcaps, numcaps, nodes = (
                res["totalcap"],
                res["allcaps"],
                res["numcaps"],
                res["nodes"],
            )
            i = look_up(float(res["c"])) - 1
            j = look_up(float(res["mod"]))
            threshold = look_up_threshold(int(res["N"]))
            vec = cap2vec(allcaps, maxdel=maxdel, maxdeg=maxdeg, threshold=threshold)
            axs[i][j].set_ylim(0, 25)
            axs[i][j].set_xlim(0, 10)
            title = "mod: {}, c: {}, r_bg: {}".format(res["mod"], res["c"], res["r_bg"])
            axs[i][j].set_title(title)
            if not i == 3:
                axs[i][j].xaxis.set_ticks(np.arange(0, 10, 1), labels=[" ", " ", " ", " ", " ", " ", " ", " ", " ", " "])
            else: 
                axs[i][j].xaxis.set_ticks(np.arange(0, 10, 1))
                
            if not j == 0:
                axs[i][j].yaxis.set_ticks(np.arange(0, 25, 5), labels=[" ", " ", " ", " ", " "])
            else:
                axs[i][j].yaxis.set_ticks(np.arange(0, 25, 5))
            
            bottom = np.zeros(len(delrange))
            values = None
            color_array = ['r', 'b', 'm', 'y', 'g']
            for k in range(len(vec[0]) - 1):
                values = vec[:, i][delrange - 1] #-1 ?
                breakpoint()
                color = color_array[k%5]
                axs[i][j].bar(delrange, values, label=f"O({k+1})", bottom=bottom, color=color)
                bottom += values
                
        plt.show()
        
    #############################################
    ##          heatmap 4D plotting            ##
    #############################################
    
    elif argv[1] == "heatmap_4D":
    
        fig = plt.figure()
        subfigs = fig.subfigures(2,1, height_ratios=[1,3])
        axs0 = subfigs[0].subplots(1,1)
        axs1 = subfigs[1].subplots(5,6)
        fig.suptitle("n" + argv[1] + "_" + argv[2], fontsize=20) #usage: #neurons value_to_plot
    
        r_start = "7.0"
        r_end = "14.0"
    
        if argv[1] == "2500":
            r_start = "7.0"
            r_end = "14.0"
        elif argv[1] == "5000":
            r_start = "6.0"
            r_end = "13.0"
        elif argv[1] == "10000":
            r_start = "5.0"
            r_end = "12.0"
     
        #usage: name_of_src_folder
        path_default = "/home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/" + "n" + argv[1] + "_scans" + "/default_full_scan_mod0.0_c0.2/res/"
        path_detail = "/home/noah/Thesis_Git/clustered_memory_network/output/jureca_scans/" + "n" + argv[1] + "_scans" + "/detail_scans_r" + r_start + "-" + r_end + "_g5.0-32.0"
    
        coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    
        name = argv[2] #column name of data to plot
      
        pretty_heatmap(
                    path_default,
                    ax=axs0,
                    name=name,
                    cmap=coolwarm,
                    y="r_bg",
                    x="g",
                    title="n" + argv[1] + "_" + argv[2] + "_default_full_scan_mod0.0_c0.2",
                    fontsize=18,
                    cmap_lim=[None,None],
                    cbar_shrink=0.7,
                )
        mod_array = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        c_array = ["0.2", "0.4", "0.6", "0.8", "1.0"]
        cmap_lim = [None,None]
        for i in range (0,5):
            for j in range (0,6):
                
                mod = mod_array[j]
                c = c_array[i]
                path_end = "/scan_n" + argv[1] + "_r" + r_start + "-" + r_end + "_step1.0_g5.0-32.0_step1.0_mod" + mod + "_c" + c + "/res/"
                title = "mod" + mod + "_c" + c
                    
                pretty_heatmap(
                    path_detail + path_end,
                    ax=axs1[i][j],
                    name=name,
                    cmap=coolwarm,
                    y="r_bg",
                    x="g",
                    title=title,
                    fontsize=18,
                    cmap_lim=cmap_lim,
                    cbar_shrink=0.7,
                )
            
        plt.show()
