import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button

molar_mass_tritium = 3.016  # g/mol
specific_activity_tritium = 3.57e14  # Bq/g
mol_to_activity_tritium = molar_mass_tritium * specific_activity_tritium  # Bq/mol
sec_to_hour = 1 / 3600

EPS = 1e-20


class ConcentrationAnimator:
    """Interactive animation for concentration profiles over time."""

    def __init__(
        self,
        times,
        c_T_solutions,
        y_T2_solutions,
        x_ct,
        x_y,
        inventories_T_salt=None,
        source_T=None,
        flux_T=None,
        show_activity=False,
        figsize=None,
        hspace=0.4,
    ):
        """
        Initialize the animator with solution data.

        Parameters:
        -----------
        times : array_like
            Time points
        c_T_solutions : array_like
            Total concentration solutions at each time
        y_T2_solutions : array_like
            Y-component concentration solutions at each time
        x_ct : array_like
            Spatial coordinates for total concentration
        x_y : array_like
            Spatial coordinates for y-component concentration
        inventories_T_salt : array_like, optional
            Integrated concentration values over time
        source_T : array_like, optional
            Tritium source term over time [mol/s]
        flux_T : array_like, optional
            Tritium extraction flux over time [mol/s]
        figsize : tuple, optional
            Figure size as (width, height) in inches
        hspace : float, optional
            Vertical spacing between subplots
        """
        self.times_hr = np.array(times) * sec_to_hour
        self.c_T_solutions = np.array(c_T_solutions)
        self.y_T2_solutions = np.array(y_T2_solutions)
        self.x_ct = x_ct
        self.x_y = x_y
        self.inventories_T_salt = inventories_T_salt
        self.source_T = None if source_T is None else np.array(source_T)
        self.flux_T = None if flux_T is None else np.array(flux_T)
        self.show_activity = show_activity
        self.figsize = figsize
        self.hspace = hspace

        if self.inventories_T_salt is not None and self.show_activity:
            self.inventories_T_salt_display = (
                np.array(self.inventories_T_salt) * mol_to_activity_tritium
            )
        else:
            self.inventories_T_salt_display = self.inventories_T_salt

        if (
            self.source_T is not None
            and self.source_T.shape[0] != self.times_hr.shape[0]
        ):
            raise ValueError("source_T must have the same length as times")
        if self.flux_T is not None and self.flux_T.shape[0] != self.times_hr.shape[0]:
            raise ValueError("flux_T must have the same length as times")

        # Animation state
        self.is_animating = False
        self.animation_timer = None

        self._setup_plot()
        self._setup_slider()
        self._setup_animation_button()

    def _setup_plot(self):
        """Setup the initial plot with subplots."""
        nrows = 3 if self.inventories_T_salt is not None else 2
        default_figsize = (11, 8) if nrows == 3 else (11, 6.3)
        self.fig = plt.figure(figsize=self.figsize or default_figsize)
        gs = gridspec.GridSpec(nrows, 1, figure=self.fig, hspace=self.hspace)

        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1], sharex=ax1)
        self.axs = [ax1, ax2]

        if self.inventories_T_salt is not None:
            # Third axis intentionally has an independent x-scale (time).
            ax3 = self.fig.add_subplot(gs[2])
            self.axs.append(ax3)

        # Balance horizontal margins so the plot area is visually centered.
        plt.subplots_adjust(left=0.12, right=0.92, top=0.94, bottom=0.125)

        # Create initial plots
        (self.line1,) = self.axs[0].plot(
            self.x_ct, self.c_T_solutions[0], "b-", linewidth=2
        )
        (self.line2,) = self.axs[1].plot(
            self.x_y, self.y_T2_solutions[0], "r-", linewidth=2
        )
        if self.inventories_T_salt is not None:
            (self.line3,) = self.axs[2].plot(
                self.times_hr, self.inventories_T_salt_display, "g-", linewidth=2
            )
            (self.time_marker,) = self.axs[2].plot(
                [self.times_hr[0]],
                [self.inventories_T_salt_display[0]],
                "ko",
                markersize=6,
            )

            self.ax3_secondary = None
            self.source_line = None
            self.flux_line = None
            self.source_marker = None
            self.flux_marker = None
            secondary_lines = []
            secondary_labels = []
            if self.source_T is not None or self.flux_T is not None:
                self.ax3_secondary = self.axs[2].twinx()
                if self.source_T is not None:
                    (self.source_line,) = self.ax3_secondary.plot(
                        self.times_hr,
                        self.source_T,
                        color="tab:orange",
                        linestyle=":",
                        linewidth=1.8,
                    )
                    (self.source_marker,) = self.ax3_secondary.plot(
                        [self.times_hr[0]],
                        [self.source_T[0]],
                        marker="o",
                        color="tab:orange",
                        markersize=5,
                        linestyle="None",
                    )
                    secondary_lines.append(self.source_line)
                    secondary_labels.append(r"$S_T$")
                if self.flux_T is not None:
                    (self.flux_line,) = self.ax3_secondary.plot(
                        self.times_hr,
                        self.flux_T,
                        color="magenta",
                        linestyle="dashdot",
                        linewidth=1.8,
                    )
                    (self.flux_marker,) = self.ax3_secondary.plot(
                        [self.times_hr[0]],
                        [self.flux_T[0]],
                        marker="s",
                        color="magenta",
                        markersize=5,
                        linestyle="None",
                    )
                    secondary_lines.append(self.flux_line)
                    secondary_labels.append(r"$\Phi_T$")

                self.ax3_secondary.set_ylabel("Source / Flux [mol/s]")
                self.ax3_secondary.grid(False)

                sec_vals = []
                if self.source_T is not None:
                    sec_vals.append(self.source_T)
                if self.flux_T is not None:
                    sec_vals.append(self.flux_T)
                sec_vals = np.concatenate(sec_vals)
                sec_min = np.min(sec_vals)
                sec_max = np.max(sec_vals)
                self.ax3_secondary.set_ylim(
                    (sec_min - EPS) * 0.9, (sec_max + EPS) * 1.1
                )
                # if np.isclose(sec_min, sec_max):
                #     pad = max(1.0, abs(sec_max) * 0.1)
                #     self.ax3_secondary.set_ylim(sec_min - pad, sec_max + pad)
                # else:
                #     pad = 0.1 * (sec_max - sec_min)
                #     self.ax3_secondary.set_ylim(sec_min - pad, sec_max + pad)

                primary_lines = [self.line3]
                primary_labels = ["Inventory"]
                self.axs[2].legend(
                    primary_lines + secondary_lines,
                    primary_labels + secondary_labels,
                    loc="best",
                )

        # Setup axes properties
        self.axs[0].set_ylabel(r"$c_T \: [mol/m^3]$")
        self.axs[0].set_title(
            f"Concentration profile in breeder at t={self.times_hr[0]:.1f} hr"
        )
        self.axs[0].grid(True, alpha=0.3)
        self.axs[0].set_ylim(
            (self.c_T_solutions.min() - EPS) * 0.9,
            (self.c_T_solutions.max() + EPS) * 1.1,
        )

        self.axs[1].set_ylabel(r"$y_{T2} \: [-]$")
        self.axs[1].set_xlabel("Position along tank height [m]")
        self.axs[1].set_title(
            f"T fraction in sparging gas at t={self.times_hr[0]:.1f} hr"
        )
        self.axs[1].grid(True, alpha=0.3)
        self.axs[1].set_ylim(
            (self.y_T2_solutions.min() - EPS) * 0.9,
            (self.y_T2_solutions.max() + EPS) * 1.1,
        )

        if self.inventories_T_salt is not None:
            if self.show_activity:
                self.axs[2].set_ylabel(r"$A_T \: [Bq]$")
                self.axs[2].set_title("Total T activity in breeder [Bq]")
            else:
                self.axs[2].set_ylabel(r"$n_T \: [mol]$")
                self.axs[2].set_title("Total T quantity in breeder [mol]")
            self.axs[2].set_xlabel("Time [hours]")
            self.axs[2].grid(True, alpha=0.3)
            self.axs[2].set_ylim(
                (self.inventories_T_salt_display.min() - EPS) * 0.9,
                (self.inventories_T_salt_display.max() + EPS) * 1.1,
            )

    def _setup_slider(self):
        """Setup the time slider."""
        ax_slider = plt.axes([0.2, 0.03, 0.55, 0.04])
        self.time_slider = Slider(
            ax_slider,
            "Time (hr)",
            self.times_hr.min(),
            self.times_hr.max(),
            valinit=self.times_hr[0],
            valfmt="%.1f",
        )
        self.time_slider.on_changed(self._update_plot)

    def _setup_animation_button(self):
        """Setup the animation toggle button."""
        ax_button = plt.axes([0.8, 0.03, 0.1, 0.035])
        self.anim_button = Button(ax_button, "Animate")
        self.anim_button.on_clicked(self._animate_toggle)

    def _update_plot(self, val):
        """Update the plots based on slider value."""
        # Find the closest time index
        current_time = self.time_slider.val
        idx = np.argmin(np.abs(self.times_hr - current_time))

        # Update the plots
        self.line1.set_ydata(self.c_T_solutions[idx])
        self.line2.set_ydata(self.y_T2_solutions[idx])

        # Update titles
        self.axs[0].set_title(
            f"Concentration profile in breeder at t={self.times_hr[idx]:.1f} hr"
        )
        self.axs[1].set_title(
            f"T fraction in sparging gas at t={self.times_hr[idx]:.1f} hr"
        )
        if self.inventories_T_salt is not None:
            self.time_marker.set_data(
                [self.times_hr[idx]], [self.inventories_T_salt_display[idx]]
            )
            if self.source_marker is not None:
                self.source_marker.set_data([self.times_hr[idx]], [self.source_T[idx]])
            if self.flux_marker is not None:
                self.flux_marker.set_data([self.times_hr[idx]], [self.flux_T[idx]])

        self.fig.canvas.draw_idle()

    def _animate_toggle(self, event):
        """Toggle animation on/off."""
        if not self.is_animating:
            self.is_animating = True
            self.anim_button.label.set_text("Stop")
            self._start_animation()
        else:
            self.is_animating = False
            self.anim_button.label.set_text("Animate")
            if self.animation_timer:
                self.animation_timer.stop()

    def _start_animation(self):
        """Start the animation loop."""

        def animate_step():
            if not self.is_animating:
                return

            current_val = self.time_slider.val
            next_val = current_val + (self.times_hr.max() - self.times_hr.min()) / 50

            if next_val > self.times_hr.max():
                next_val = self.times_hr.min()

            self.time_slider.set_val(next_val)

        self.animation_timer = self.fig.canvas.new_timer(interval=100)
        self.animation_timer.add_callback(animate_step)
        self.animation_timer.start()

    def show(self):
        """Display the interactive plot."""
        plt.show()


def create_animation(
    times,
    c_T_solutions,
    y_T2_solutions,
    x_ct,
    x_y,
    c_integrated=None,
    source_T=None,
    flux_T=None,
    show_activity=False,
    figsize=None,
    hspace=0.4,
):
    """
    Convenience function to create and show animation.

    Parameters:
    -----------
    times : array_like
        Time points
    c_T_solutions : array_like
        Total concentration solutions at each time
    y_T2_solutions : array_like
        Y-component concentration solutions at each time
    x_ct : array_like
        Spatial coordinates for total concentration
    x_y : array_like
        Spatial coordinates for y-component concentration
    c_integrated : array_like, optional
        Integrated concentration values over time
    source_T : array_like, optional
        Tritium source term over time [mol/s]
    flux_T : array_like, optional
        Tritium extraction flux over time [mol/s]
    show_activity : bool, optional
        If True, convert integrated tritium amount from mol to activity in Bq
    figsize : tuple, optional
        Figure size as (width, height) in inches
    hspace : float, optional
        Vertical spacing between subplots

    Returns:
    --------
    ConcentrationAnimator
        The animator instance
    """
    animator = ConcentrationAnimator(
        times,
        c_T_solutions,
        y_T2_solutions,
        x_ct,
        x_y,
        inventories_T_salt=c_integrated,
        source_T=source_T,
        flux_T=flux_T,
        show_activity=show_activity,
        figsize=figsize,
        hspace=hspace,
    )
    animator.show()
    return animator
