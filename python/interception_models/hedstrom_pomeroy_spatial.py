def main():
    """
    calculate the spearman rank of raster snow metric with spatial interception estimates from hedstrom & pomeroy 1998
    calculated from raster canopy metrics.
    :return:
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt


    # config
    plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\modeling snow accumulation\\hp\\"

    # canopy structure
    df_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
    df = pd.read_csv(df_in)

    lai = df.loc[:, "lrs_lai_2000"]
    # lai = df.loc[:, "lrs_lai_60_deg"]
    # lai = df.loc[:, "lrs_lai_15_deg"]
    # lai = df.loc[:, "lrs_lai_1_deg"]
    cc = df.loc[:, "lrs_cc"]
    chm = df.loc[:, "chm_25"]

    # constants (if testing for monotonicity, do these values matter?)
    snow_dens = 68  # kg/m^3
    s_bar = 5.9  # kg/m^2
    h_bar = np.nanmax(chm)
    l_0 = 0

    ###
    # def spr_from_hp(precip=precip, snow_dens=snow_dens, s_bar=s_bar, h_bar=h_bar, l_0 = l_0):
    ss = s_bar * (0.27 + 46 / snow_dens)
    l_star = ss * lai
    cp = cc / (1 - (cc * chm / h_bar))
    kk = cp / l_star
    ##



    # correlations
    precip = 105.06450584
    swevar = df.loc[:, "swe_fcon_19_052"]
    interception = (l_star - l_0) * (1 - np.exp(-kk * precip))
    throughfall = precip - interception
    valid = ~np.isnan(throughfall) & ~np.isnan(swevar)
    spearman_swe = spearmanr(swevar[valid], throughfall[valid])

    fig = plt.figure()
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
    ax1 = fig.add_subplot(111)
    ax1.set_title('SWE vs. Hedstrom-Pomeroy throughfall $F_{HP}$\n Forest, 21 Feb. 2019')
    ax1.set_xlabel("SWE [mm]")
    ax1.set_ylabel("$F_{HP}$ [mm]")
    plt.ylim(np.nanquantile(throughfall, .005), np.nanquantile(throughfall, .995))
    plt.xlim(np.nanquantile(swevar, .005), np.nanquantile(swevar, .995))
    plt.scatter(swevar[valid], throughfall[valid], alpha=0.10, s=2)
    mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
    # plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
    fig.savefig(plot_out_dir + "SWE vs HP 19_052.png")

    precip = 3  # mm swe
    swevar = df.loc[:, "dswe_fnsd_19_045-19_050"]
    interception = (l_star - l_0) * (1 - np.exp(-kk * precip))
    throughfall = precip - interception
    valid = ~np.isnan(throughfall) & ~np.isnan(swevar)
    spearman_storm_1 = spearmanr(swevar[valid], throughfall[valid])

    fig = plt.figure()
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
    ax1 = fig.add_subplot(111)
    ax1.set_title('$\Delta$SWE vs. Hedstrom-Pomeroy throughfall $F_{HP}$\n Forest, 14-19 Feb. 2019')
    ax1.set_xlabel("$\Delta$SWE [mm]")
    ax1.set_ylabel("$F_{HP}$ [mm]")
    plt.ylim(np.nanquantile(throughfall, .005), np.nanquantile(throughfall, .995))
    plt.xlim(np.nanquantile(swevar, .005), np.nanquantile(swevar, .995))
    plt.scatter(swevar[valid], throughfall[valid], alpha=0.05, s=2)
    mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
    # plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
    fig.savefig(plot_out_dir + "SWE vs HP 045-050.png")

    precip = 8.6  # mm swe
    swevar = df.loc[:, "dswe_fnsd_19_050-19_052"]
    interception = (l_star - l_0) * (1 - np.exp(-kk * precip))
    throughfall = precip - interception
    valid = ~np.isnan(throughfall) & ~np.isnan(swevar)
    spearman_storm_2 = spearmanr(swevar[valid], throughfall[valid])

    fig = plt.figure()
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
    ax1 = fig.add_subplot(111)
    ax1.set_title('$\Delta$SWE vs. Hedstrom-Pomeroy throughfall $F_{HP}$\n Forest, 19-21 Feb. 2019')
    ax1.set_xlabel("$\Delta$SWE [mm]")
    ax1.set_ylabel("$F_{HP}$ [mm]")
    plt.ylim(np.nanquantile(throughfall, .005), np.nanquantile(throughfall, .995))
    plt.xlim(np.nanquantile(swevar, .005), np.nanquantile(swevar, .995))
    plt.scatter(swevar[valid], throughfall[valid], alpha=0.05, s=2)
    mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
    # plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
    fig.savefig(plot_out_dir + "SWE vs HP 050-052.png")

        # return (spearman_swe, spearman_storm_1, spearman_storm_2)


    def sensitivity_test(p_range):
        output = np.zeros((len(p_range), 2))
        for ii in range(0, len(p_range)):
            output[ii, :] = spr_from_hp(precip=p_range[ii])  # low sensitivity -- greater correlations for greater precip values (up to ~50) -- use observed values though
            # output[ii, :] = spr_from_hp(snow_dens=p_range[ii])  # no sensitivity -- use observed value
            # output[ii, :] = spr_from_hp(s_bar=p_range[ii])  # low sensitivity -- use textbook value
            # output[ii, :] = spr_from_hp(h_bar=p_range[ii])  # high sensitivity -- less sensitive for values around max canopy height
            # output[ii, :] = spr_from_hp(l_0=p_range[ii])  # moderate sensitivity -- greatest correlation around 0 -- use 0

        return output

    # p_range = np.linspace(0, 40, 50)
    # out = sensitivity_test(p_range)
    #
    # plt.plot(p_range, out[:, 0])
    # plt.plot(p_range, out[:, 1])

    # spr_from_hp()

    # optimization
    from scipy.optimize import fmin_bfgs, minimize

    # valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_050-19_052"])
    # swevar = df.loc[:, "swe_fcon_19_052"]
    # swevar = df.loc[:, "dswe_fnsd_19_045-19_050"]
    swevar = df.loc[:, "dswe_fnsd_19_050-19_052"]
    valid = ~np.isnan(swevar)

    def hp_accum(x0):
        # s_bar, h_bar, l_0, snow_dens, precip = x0
        s_bar = 5.9
        snow_dens = 68
        h_bar = np.nanmax(chm)
        l_0, precip = x0
        ss = s_bar * (0.27 + 46 / snow_dens)
        l_star = ss * lai
        cp = cc / (1 - (cc * chm / h_bar))
        kk = cp / l_star
        interception = (l_star - l_0) * (1 - np.exp(-kk * precip))
        accumulation = precip - interception

        ssres = np.sum((swevar - accumulation) ** 2)

        return ssres


    def rsq(x0):
        ssres = hp_accum(x0)

        # dswe = covariant
        dswe = swevar[valid]

        sstot = np.sum((dswe - np.mean(dswe)) ** 2)
        return 1 - ssres / sstot

    def hp_accum_data(x0):
        # s_bar, h_bar, l_0, snow_dens, precip = x0
        s_bar = 5.9
        snow_dens = 68
        h_bar = np.nanmax(chm)
        l_0, precip = x0
        ss = s_bar * (0.27 + 46 / snow_dens)
        l_star = ss * lai
        cp = cc / (1 - (cc * chm / h_bar))
        kk = cp / l_star
        interception = (l_star - l_0) * (1 - np.exp(-kk * precip))
        accumulation = precip - interception

        # ssres = np.sum((swevar - accumulation) ** 2)

        return accumulation

    #
    # Nfeval = 1
    # def callbackF(Xi):
    #     global Nfeval
    #     print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1],  Xi[2], Xi[3], Xi[4], hp_accum(Xi), rsq(Xi)))
    #     Nfeval += 1
    #
    # print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}   {7:9s}'.format('Iter', 's_bar', 'h_bar', 'l_0', 'snow_dens', 'precip', 'f(X)', 'R2'))
    #
    # x0 = np.array([5.9, 30, 3, 68, 100])
    #
    # [popt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = fmin_bfgs(hp_accum, x0, callback=callbackF, maxiter=200, full_output=True, retall=False)


    res = minimize(hp_accum, (0, 3), method='L-BFGS-B', bounds=((0.0, 100.0), (0.0, 1000.0)) ,options={'maxiter': 1000})

    rsq(res.x)

    throughfall = hp_accum_data(res.x)
    plt.scatter(swevar, throughfall, s=2, alpha=0.25)

if __name__ == "__main__":
    main()