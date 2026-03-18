import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button

molar_mass_tritium = 3.016  # g/mol
specific_activity_tritium = 3.57e14  # Bq/g
mol_to_activity_tritium = molar_mass_tritium * specific_activity_tritium  # Bq/mol


class ConcentrationAnimator:
    """Interactive animation for concentration profiles over time."""

    def __init__(
        self,
        times,
        c_T_solutions,
        y_T2_solutions,
        x_ct,
        x_y,
        c_integrated=None,
        show_activity=False,
        figsize=None,
        hspace=0.45,
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
        c_integrated : array_like, optional
            Integrated concentration values over time
        figsize : tuple, optional
            Figure size as (width, height) in inches
        hspace : float, optional
            Vertical spacing between subplots
        """
        self.times = np.array(times)
        self.c_T_solutions = np.array(c_T_solutions)
        self.y_T2_solutions = np.array(y_T2_solutions)
        self.x_ct = x_ct
        self.x_y = x_y
        self.c_integrated = c_integrated
        self.show_activity = show_activity
        self.figsize = figsize
        self.hspace = hspace

        if self.c_integrated is not None and self.show_activity:
            self.c_integrated_display = (
                np.array(self.c_integrated) * mol_to_activity_tritium
            )
        else:
            self.c_integrated_display = self.c_integrated

        # Animation state
        self.is_animating = False
        self.animation_timer = None

        self._setup_plot()
        self._setup_slider()
        self._setup_animation_button()

    def _setup_plot(self):
        """Setup the initial plot with subplots."""
        nrows = 3 if self.c_integrated is not None else 2
        default_figsize = (12, 9.5) if nrows == 3 else (11, 7)
        self.fig = plt.figure(figsize=self.figsize or default_figsize)
        gs = gridspec.GridSpec(nrows, 1, figure=self.fig, hspace=self.hspace)

        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1], sharex=ax1)
        self.axs = [ax1, ax2]

        if self.c_integrated is not None:
            # Third axis intentionally has an independent x-scale (time).
            ax3 = self.fig.add_subplot(gs[2])
            self.axs.append(ax3)

        # Leave room for slider/button and avoid subplot title/label overlap.
        plt.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.2)

        # Create initial plots
        (self.line1,) = self.axs[0].plot(
            self.x_ct, self.c_T_solutions[0], "b-", linewidth=2
        )
        (self.line2,) = self.axs[1].plot(
            self.x_y, self.y_T2_solutions[0], "r-", linewidth=2
        )
        if self.c_integrated is not None:
            (self.line3,) = self.axs[2].plot(
                self.times, self.c_integrated_display, "g-", linewidth=2
            )
            (self.time_marker,) = self.axs[2].plot(
                [self.times[0]], [self.c_integrated_display[0]], "ko", markersize=6
            )

        # Setup axes properties
        self.axs[0].set_ylabel(r"$c_T \: [mol/m^3]$")
        self.axs[0].set_title(
            f"Concentration profile in breeder at t={self.times[0]:.1f}s"
        )
        self.axs[0].grid(True, alpha=0.3)
        self.axs[0].set_ylim(
            self.c_T_solutions.min() * 0.9, self.c_T_solutions.max() * 1.1
        )

        self.axs[1].set_ylabel(r"$y_{T2} \: [-]$")
        self.axs[1].set_xlabel("Position along tank height [m]")
        self.axs[1].set_title(f"T fraction in sparging gas at t={self.times[0]:.1f}s")
        self.axs[1].grid(True, alpha=0.3)
        self.axs[1].set_ylim(
            self.y_T2_solutions.min() * 0.9, self.y_T2_solutions.max() * 1.1
        )

        if self.c_integrated is not None:
            if self.show_activity:
                self.axs[2].set_ylabel(r"$A_T \: [Bq]$")
                self.axs[2].set_title("Total T activity in breeder [Bq]")
            else:
                self.axs[2].set_ylabel(r"$n_T \: [mol]$")
                self.axs[2].set_title("Total T quantity in breeder [mol]")
            self.axs[2].set_xlabel("Time (t)")
            self.axs[2].grid(True, alpha=0.3)
            self.axs[2].set_ylim(
                self.c_integrated_display.min() * 0.9,
                self.c_integrated_display.max() * 1.1,
            )

    def _setup_slider(self):
        """Setup the time slider."""
        ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
        self.time_slider = Slider(
            ax_slider,
            "Time (s)",
            self.times.min(),
            self.times.max(),
            valinit=self.times[0],
            valfmt="%.1f",
        )
        self.time_slider.on_changed(self._update_plot)

    def _setup_animation_button(self):
        """Setup the animation toggle button."""
        ax_button = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.anim_button = Button(ax_button, "Animate")
        self.anim_button.on_clicked(self._animate_toggle)

    def _update_plot(self, val):
        """Update the plots based on slider value."""
        # Find the closest time index
        current_time = self.time_slider.val
        idx = np.argmin(np.abs(self.times - current_time))

        # Update the plots
        self.line1.set_ydata(self.c_T_solutions[idx])
        self.line2.set_ydata(self.y_T2_solutions[idx])

        # Update titles
        self.axs[0].set_title(
            f"Concentration profile in breeder at t={self.times[idx]:.1f}s"
        )
        self.axs[1].set_title(f"T fraction in sparging gas at t={self.times[idx]:.1f}s")
        if self.c_integrated is not None:
            self.time_marker.set_data(
                [self.times[idx]], [self.c_integrated_display[idx]]
            )

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
            next_val = current_val + (self.times.max() - self.times.min()) / 50

            if next_val > self.times.max():
                next_val = self.times.min()

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
    show_activity=False,
    figsize=None,
    hspace=0.35,
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
        c_integrated=c_integrated,
        show_activity=show_activity,
        figsize=figsize,
        hspace=hspace,
    )
    animator.show()
    return animator
