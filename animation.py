import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class ConcentrationAnimator:
    """Interactive animation for concentration profiles over time."""

    def __init__(self, times, c_T_solutions, y_T2_solutions, x_ct, x_y):
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
        """
        self.times = np.array(times)
        self.c_T_solutions = np.array(c_T_solutions)
        self.y_T2_solutions = np.array(y_T2_solutions)
        self.x_ct = x_ct
        self.x_y = x_y

        # Animation state
        self.is_animating = False
        self.animation_timer = None

        self._setup_plot()
        self._setup_slider()
        self._setup_animation_button()

    def _setup_plot(self):
        """Setup the initial plot with subplots."""
        self.fig, self.axs = plt.subplots(2, sharex=True, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for controls

        # Create initial plots
        (self.line1,) = self.axs[0].plot(
            self.x_ct, self.c_T_solutions[0], "b-", linewidth=2
        )
        (self.line2,) = self.axs[1].plot(
            self.x_y, self.y_T2_solutions[0], "r-", linewidth=2
        )

        # Setup axes properties
        self.axs[0].set_ylabel("Total Concentration c_T")
        self.axs[0].set_title(f"Total Concentration at t={self.times[0]:.1f}s")
        self.axs[0].grid(True, alpha=0.3)
        self.axs[0].set_ylim(
            self.c_T_solutions.min() * 0.9, self.c_T_solutions.max() * 1.1
        )

        self.axs[1].set_ylabel("Y-Component Concentration y_T2")
        self.axs[1].set_xlabel("Position (x)")
        self.axs[1].set_title(f"Y-Component Concentration at t={self.times[0]:.1f}s")
        self.axs[1].grid(True, alpha=0.3)
        self.axs[1].set_ylim(
            self.y_T2_solutions.min() * 0.9, self.y_T2_solutions.max() * 1.1
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
        self.axs[0].set_title(f"Total Concentration at t={self.times[idx]:.1f}s")
        self.axs[1].set_title(f"Y-Component Concentration at t={self.times[idx]:.1f}s")

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


def create_animation(times, c_T_solutions, y_T2_solutions, x_ct, x_y):
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

    Returns:
    --------
    ConcentrationAnimator
        The animator instance
    """
    animator = ConcentrationAnimator(times, c_T_solutions, y_T2_solutions, x_ct, x_y)
    animator.show()
    return animator
