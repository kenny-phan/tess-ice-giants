import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.timeseries import LombScargle


def get_summed_fluxes(reduced_image_stack_pos, positions_all, r_aperture):

    summed_fluxes_all = np.array([])

    for frame_num in range(0, len(reduced_image_stack_pos[0, 0])):

        data_frame = reduced_image_stack_pos[:, :, frame_num]

        aperture = CircularAperture(positions_all[frame_num], r=r_aperture)  # 5)#1.5)
        phot_table = aperture_photometry(data_frame, aperture)
        summed_flux = phot_table["aperture_sum"].value[0]

        summed_fluxes_all = np.append(summed_fluxes_all, summed_flux)

    return summed_fluxes_all


def plot_lightcurve(target_name, root_dir, times, flux):

    print("plotting light curve ... ")

    range_flux = np.max(flux) - np.min(flux)
    min_flux = np.min(flux) - (0.05 * range_flux)
    max_flux = np.max(flux) + (0.05 * range_flux)

    plt.clf()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.scatter(times, flux, s=2, color="r")

    plt.xlim(np.min(times), np.max(times))
    plt.ylim(min_flux, max_flux)

    plt.xlabel("Times (JD)")
    plt.ylabel("Fluxes (e/s)")

    plt.tight_layout()
    plt.savefig(root_dir + "lc_%s.png" % (target_name), dpi=500)
    plt.clf()
    plt.close()

    print("saved light curve, lc_shift_diagnostic_%s.png" % (target_name))


def show_stack_frame(data_stack, root_dir, filename_ext, vmin=0, vmax=1000, framenum=0):

    plt.clf()
    plt.close()
    plt.figure(figsize=(5, 5.5))
    plt.imshow(data_stack[:, :, framenum], vmin=vmin, vmax=vmax)
    plt.colorbar(label="flux (e/s)")
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")

    plt.tight_layout()
    plt.savefig(root_dir + "snapshot_%s.png" % (filename_ext), dpi=800)
    plt.clf()
    plt.close()

    print("saved image, snapshot_%s.png" % (filename_ext))


def plot_shift_lightcurve_diagnostic(
    target_name,
    root_dir,
    subtracted_image_stack,
    times_vetted,
    positions_vetted,
    r_aperture,
    full_scale=False,
):

    positions_shifted_ahead = np.copy(positions_vetted)
    positions_shifted_ahead[:, 1] -= 20

    positions_shifted_behind = np.copy(positions_vetted)
    positions_shifted_behind[:, 1] += 20

    positions_shifted_side = np.copy(positions_vetted)
    positions_shifted_side[:, 0] += 20

    positions_shifted_side2 = np.copy(positions_vetted)
    positions_shifted_side2[:, 0] -= 20

    lc_shifted_ahead = get_summed_fluxes(
        subtracted_image_stack, positions_shifted_ahead, r_aperture
    )
    lc_shifted_behind = get_summed_fluxes(
        subtracted_image_stack, positions_shifted_behind, r_aperture
    )
    lc_shifted_side = get_summed_fluxes(
        subtracted_image_stack, positions_shifted_side, r_aperture
    )
    lc_shifted_side2 = get_summed_fluxes(
        subtracted_image_stack, positions_shifted_side2, r_aperture
    )

    # plot light curves, shifted
    plt.clf()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.scatter(
        times_vetted, lc_shifted_ahead, s=2, color="purple", label="shifted +20 ahead"
    )
    plt.scatter(
        times_vetted, lc_shifted_behind, s=2, color="r", label="shifted -20 behind"
    )
    plt.scatter(
        times_vetted, lc_shifted_side, s=2, color="b", label="shifted +20 to side"
    )
    plt.scatter(
        times_vetted, lc_shifted_side2, s=2, color="k", label="shifted -20 to side"
    )
    plt.legend()

    plt.xlim(np.min(times_vetted), np.max(times_vetted))

    if full_scale == True:
        flux = np.concatenate((lc_shifted_ahead, lc_shifted_behind, lc_shifted_side, lc_shifted_side2))
        flux = flux[~np.isnan(flux)]

        range_flux = np.max(flux) - np.min(flux)
        min_flux = np.min(flux) - (0.05 * range_flux)
        max_flux = np.max(flux) + (0.05 * range_flux)

        plt.ylim(min_flux, max_flux)

    else:
        plt.ylim(-1000, 1000)

    plt.xlabel("Times (JD)")
    plt.ylabel("Fluxes (e/s)")

    plt.tight_layout()
    plt.savefig(root_dir + "lc_shift_diagnostic_%s.png" % (target_name), dpi=500)
    plt.clf()
    plt.close()

    print("saved light curve diagnostic, lc_shift_diagnostic_%s.png" % (target_name))

    # plot Lomb-Scargle periodograms associated with each of the shifted light curves
    frequency = np.linspace(0, 2, 5000)
    power_ls_shifted_ahead = LombScargle(times_vetted, lc_shifted_ahead).power(frequency)
    power_ls_shifted_behind = LombScargle(times_vetted, lc_shifted_behind).power(frequency)
    power_ls_shifted_side = LombScargle(times_vetted, lc_shifted_side).power(frequency)
    power_ls_shifted_side2 = LombScargle(times_vetted, lc_shifted_side2).power(frequency)

    plt.figure(figsize=(8, 4))
    plt.plot(frequency, power_ls_shifted_ahead, color='purple', label="shifted +20 ahead")
    plt.plot(frequency, power_ls_shifted_behind, color='r', label="shifted -20 behind")
    plt.plot(frequency, power_ls_shifted_side, color='b', label="shifted +20 to side")
    plt.plot(frequency, power_ls_shifted_side2, color='k', label="shifted -20 to side")
    plt.xlabel('frequency (cycles/day)')
    plt.ylabel('power')
    plt.xlim(np.min(frequency), np.max(frequency))
    plt.legend()
    plt.savefig(root_dir+'lombscargle_shift_diagnostic_%s.png' %(target_name), dpi=400)
    plt.clf()
    plt.close()

    print("saved lomb-scargle diagnostic, lombscargle_shift_diagnostic_%s.png" % (target_name))


def star_proximity_tracker(
    target_name, root_dir, data_stack_vetted, positions_vetted, times_vetted, lc
):

    # plot how many pixels away from a
    # bright star the source is at any time.

    median_stack_time = np.median(data_stack_vetted, axis=2)

    # figure 1: show nearby star detections
    plt.clf()
    plt.close()
    plt.figure(figsize=(14, 14))
    plt.imshow(median_stack_time, vmin=0, vmax=1000)

    daofind = DAOStarFinder(fwhm=3, threshold=100)
    sources = daofind(median_stack_time)
    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    apertures = CircularAperture(positions, r=4.0)
    apertures.plot(color="red", lw=1.5, alpha=0.5)
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    plt.colorbar()
    plt.savefig(root_dir + "nearby_star_detections.png", dpi=400)

    # figure 2: calculate distance from sources and plot
    min_dist_from_stars = np.array([])
    for source_position in positions_vetted:
        dist_from_sources = np.sqrt(
            ((source_position[0] - sources["xcentroid"]) ** 2.0)
            + ((source_position[1] - sources["ycentroid"]) ** 2.0)
        )
        min_dist_from_stars = np.append(min_dist_from_stars, np.min(dist_from_sources))

    plt.clf()
    plt.close()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    ax[0].scatter(times_vetted, lc, color="purple")
    ax[0].set_ylabel("summed flux, %s (e/s)" % (target_name))

    ax[1].scatter(times_vetted, min_dist_from_stars, color="teal")
    ax[1].set_xlabel("time (JD)")
    ax[1].set_ylabel("minimum distance to a nearby star (pixels)")
    plt.savefig(root_dir + "nearby_star_proximity_tracker.png", dpi=400)

    print("completed star proximity tracker diagnostics")


def comparison_flux_nearby_star(
    target_name,
    root_dir,
    data_stack_vetted,
    times_vetted,
    lc,
    r_aperture,
    threshold=10000,
):

    # show flux of source vs. a nearby star
    daofind = DAOStarFinder(fwhm=3, threshold=threshold)

    median_stack_time = np.median(data_stack_vetted, axis=2)
    sources_all = daofind(median_stack_time)
    print(sources_all)

    for ind_star in range(0, len(sources_all)):
        sources = sources_all[ind_star]

        position_star = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = CircularAperture(position_star, r=4.0)

        # show where star is in image
        plt.clf()
        plt.close()
        plt.figure(figsize=(5, 5.5))
        plt.imshow(median_stack_time, vmin=0, vmax=1000)
        plt.colorbar(label="flux (e/s)")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        apertures.plot(color="red", lw=1.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(root_dir + "snapshot_star%i.png" % (ind_star), dpi=800)
        plt.clf()
        plt.close()

        positions_star = np.transpose(
            np.repeat(position_star[:, np.newaxis], len(times_vetted), axis=1)
        )

        # get summed fluxes of nearby star
        summed_fluxes_nearby_star = get_summed_fluxes(
            data_stack_vetted, positions_star, r_aperture=r_aperture
        )

        # get normalized comparison
        summed_fluxes_nearby_star_norm = summed_fluxes_nearby_star + (
            np.median(lc) - np.median(summed_fluxes_nearby_star)
        )

        # plot comparison
        plt.figure(figsize=(10, 5))
        plt.scatter(times_vetted, lc, color="green", label=target_name)
        plt.scatter(
            times_vetted,
            summed_fluxes_nearby_star,
            color="orange",
            label="star comparison",
        )
        plt.ylabel("summed flux (e/s)")
        plt.xlabel("times (JD)")
        plt.xlim(np.min(times_vetted), np.max(times_vetted))
        plt.legend()
        plt.savefig(root_dir + "%s_v_star%i_lc.png" % (target_name, ind_star), dpi=400)
        plt.clf()
        plt.close()

        # plot comparison
        plt.figure(figsize=(10, 5))
        plt.scatter(
            times_vetted,
            lc - summed_fluxes_nearby_star_norm,
            color="green",
            label="%s minus star %i, norm" % (target_name, ind_star),
        )
        plt.ylabel("summed flux (e/s)")
        plt.xlabel("times (JD)")
        plt.xlim(np.min(times_vetted), np.max(times_vetted))
        plt.legend()
        plt.savefig(
            root_dir + "%s_minus_star%i_lc.png" % (target_name, ind_star), dpi=400
        )
        plt.clf()
        plt.close()

        print("completed comparison flux diagnostics with nearby stars")
