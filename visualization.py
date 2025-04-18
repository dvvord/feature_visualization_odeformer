import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Function to plot solution
def plot_solution(solution, time_point, var1, var2, var3=None, var4=None, 
                  var1_name="", var2_name="", var3_name=None, var4_name=None,
                  top_activation_times=None, top_activation_values=None):
    """Plot the solution with optional highlights for activation points."""
    times = solution['time_points']
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.plot(times, var1, label=var1_name, linewidth=2)
    ax1.plot(times, var2, label=var2_name, linewidth=2)
    
    if var3 is not None and var3_name is not None:
        if var4 is not None and var4_name is not None:
            # Double pendulum case
            ax1.plot(times, var3, label=var3_name, linewidth=1.5)
            ax1.plot(times, var4, label=var4_name, linewidth=1.5, linestyle='--')
        else:
            # 3D system or something with 3 variables
            ax1.plot(times, var3, label=var3_name, linewidth=1.5, linestyle='--')
    
    # Add colorized vertical lines for top activations if available
    if top_activation_times and top_activation_values:
        min_val = min(top_activation_values)
        max_val = max(top_activation_values)
        norm = plt.Normalize(min_val, max_val)
        cmap = plt.cm.cool  # Use a visually appealing colormap
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Add slim, color-coded lines
        for i, t_idx in enumerate(top_activation_times):
            t = times[t_idx]
            color = cmap(norm(top_activation_values[i]))
            ax1.axvline(x=t, color=color, alpha=0.6, linewidth=0.8)
        
        # Add a colorbar
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('Activation Value')
        
        st.caption(f"Found {len(top_activation_times)} time points with top activations")
    
    # Highlight the selected time point
    if time_point < len(times):
        selected_time = times[time_point]
        ax1.axvline(x=selected_time, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.plot([selected_time], [var1[time_point]], 'ro', markersize=6)
        ax1.plot([selected_time], [var2[time_point]], 'ro', markersize=6)
        if var3 is not None:
            ax1.plot([selected_time], [var3[time_point]], 'ro', markersize=6)
            if var4 is not None:
                ax1.plot([selected_time], [var4[time_point]], 'ro', markersize=6)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('System Solution')
    
    return fig1

# Function to plot feature activation
def plot_feature_activation(times, feature_values, time_point, top_activation_times=None):
    """Plot feature activation over time."""
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.plot(times, feature_values)
    
    # Highlight selected time point
    selected_time = times[time_point]
    ax2.axvline(x=selected_time, color='r', linestyle='--', alpha=0.7)
    ax2.plot([selected_time], [feature_values[time_point]], 'ro', markersize=8)
    
    # Highlight top activation times if available
    if top_activation_times and len(top_activation_times) > 0:
        for t_idx in top_activation_times:
            t = times[t_idx]
            ax2.axvline(x=t, color='magenta', alpha=0.3, linewidth=1.5)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Feature Activation')
    ax2.set_title('Feature Activation Over Time')
    ax2.grid(True)
    
    return fig2

def plot_trajectories_comparison(times, actual_trajectory, predicted_trajectory, var_names=None):
    """
    Plot comparison between actual and predicted trajectories.
    
    Args:
        times: Time points array
        actual_trajectory: Actual trajectory data (shape: [time_points, dimensions])
        predicted_trajectory: Predicted trajectory data (shape: [time_points, dimensions])
        var_names: List of variable names for each dimension
        
    Returns:
        matplotlib Figure
    """
    # Determine number of dimensions
    n_dims = actual_trajectory.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))
    
    # Handle case with single dimension
    if n_dims == 1:
        axes = [axes]
    
    # Plot each dimension
    for i in range(n_dims):
        dim = i
        var_name = var_names[i] if var_names and i < len(var_names) else f"Variable {i}"
        
        # Plot actual trajectory
        axes[i].plot(times, actual_trajectory[:, dim], 
                    color=f'C{dim}', label=f'Actual', 
                    marker='o', markersize=4, alpha=0.3)
        
        # Plot predicted trajectory
        axes[i].plot(times, predicted_trajectory[:, dim], 
                    color=f'C{dim}', label=f'Predicted', 
                    linestyle='--', linewidth=2)
        
        # Add labels and grid
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(var_name)
        axes[i].set_title(f'{var_name} vs Time')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    return fig

# Function to plot phase portrait for 2D systems
def plot_phase_portrait_2d(var1, var2, var1_name, var2_name, feature_values, time_point, system_type):
    """Plot 2D phase portrait colored by feature activations."""
    fig_phase, ax_phase = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create a colormap for feature activations
    norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
    
    # Use scatter plot with color mapped to feature activation
    scatter = ax_phase.scatter(var1, var2, c=feature_values, cmap='viridis', 
                            norm=norm, s=30, alpha=0.7)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax_phase)
    cbar.set_label('Feature Activation')
    
    # Also add a line to show trajectory
    ax_phase.plot(var1, var2, 'k-', alpha=0.3, linewidth=0.8)
    
    ax_phase.set_xlabel(var1_name)
    ax_phase.set_ylabel(var2_name)
    ax_phase.grid(True)
    ax_phase.set_title(f"{system_type} Phase Portrait (colored by feature activations)")
    
    # Mark starting point
    ax_phase.plot([var1[0]], [var2[0]], 'go', markersize=8, label='Start')
    
    # Mark current time point
    if time_point < len(var1):
        ax_phase.plot([var1[time_point]], [var2[time_point]], 'ro', markersize=8, label='Current')
        
    ax_phase.legend()
    return fig_phase

# Function to plot 3D phase portrait for Lorenz system
def plot_phase_portrait_3d(var1, var2, var3, var1_name, var2_name, var3_name, 
                         feature_values, time_point, system_type="Lorenz System"):
    """Plot 3D phase portrait for systems like Lorenz."""
    fig_phase = plt.figure(figsize=(10, 8))
    ax_phase = fig_phase.add_subplot(111, projection='3d')
    
    # Create a colormap for feature activations
    norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
    
    # Plot points colored by feature activation
    scatter = ax_phase.scatter(var1, var2, var3, 
                            c=feature_values, cmap='viridis', 
                            norm=norm, s=20, alpha=0.7)
    
    # Add a colorbar
    cbar = fig_phase.colorbar(scatter, ax=ax_phase, shrink=0.6, pad=0.1)
    cbar.set_label('Feature Activation')
    
    # Also plot the 3D trajectory as a thin line
    ax_phase.plot3D(var1, var2, var3, 'k-', linewidth=0.5, alpha=0.3)
    
    # Mark starting point
    ax_phase.scatter([var1[0]], [var2[0]], [var3[0]], color='green', s=100, label='Start')
    
    # Mark current time point
    if time_point < len(var1):
        ax_phase.scatter([var1[time_point]], [var2[time_point]], [var3[time_point]], 
                        color='red', s=100, label='Current')
        
    ax_phase.set_xlabel(var1_name)
    ax_phase.set_ylabel(var2_name)
    ax_phase.set_zlabel(var3_name)
    ax_phase.legend()
    ax_phase.set_title(f"{system_type} (colored by feature activations)")
    
    return fig_phase

# Function to plot double pendulum phase portrait
def plot_double_pendulum_phase(var1, var2, var3, var4, cart_sol, time_point, feature_values):
    """Plot phase portrait for double pendulum system."""
    fig_phase = plt.figure(figsize=(15, 8))
    
    # Create a colormap for feature activations
    norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
    
    # 1. Theta1 vs Theta2
    ax1 = fig_phase.add_subplot(131)
    scatter1 = ax1.scatter(var1, var3, c=feature_values, cmap='viridis', 
                          norm=norm, s=30, alpha=0.7)
    ax1.plot(var1, var3, 'k-', alpha=0.3, linewidth=0.5)  # Add trajectory line
    ax1.set_xlabel("Angle 1 (θ₁)")
    ax1.set_ylabel("Angle 2 (θ₂)")
    ax1.grid(True)
    ax1.set_title("θ₁ vs θ₂")
    # Add colorbar
    plt.colorbar(scatter1, ax=ax1, shrink=0.7)
    ax1.plot(var1[0], var3[0], 'go', markersize=8)
    if time_point < len(var1):
        ax1.plot(var1[time_point], var3[time_point], 'ro', markersize=8)
    
    # 2. Omega1 vs Omega2
    ax2 = fig_phase.add_subplot(132)
    scatter2 = ax2.scatter(var2, var4, c=feature_values, cmap='viridis', 
                          norm=norm, s=30, alpha=0.7)
    ax2.plot(var2, var4, 'k-', alpha=0.3, linewidth=0.5)  # Add trajectory line
    ax2.set_xlabel("Angular Velocity 1 (ω₁)")
    ax2.set_ylabel("Angular Velocity 2 (ω₂)")
    ax2.grid(True)
    ax2.set_title("ω₁ vs ω₂")
    # Add colorbar
    plt.colorbar(scatter2, ax=ax2, shrink=0.7)
    ax2.plot(var2[0], var4[0], 'go', markersize=8)
    if time_point < len(var2):
        ax2.plot(var2[time_point], var4[time_point], 'ro', markersize=8)
    
    # 3. Cartesian trajectory if available
    if cart_sol is not None:
        ax3 = fig_phase.add_subplot(133)
        times = range(cart_sol.shape[1])
        
        # Plot pendulum arms
        for t in range(0, len(times), max(1, len(times)//100)):  # Plot a subset of points
            alpha = max(0.05, min(0.5, t/len(times)))
            # Draw first pendulum arm
            ax3.plot([0, cart_sol[0, t]], [0, cart_sol[1, t]], 'k-', alpha=alpha, linewidth=1)
            # Draw second pendulum arm
            ax3.plot([cart_sol[0, t], cart_sol[2, t]], [cart_sol[1, t], cart_sol[3, t]], 'k-', alpha=alpha, linewidth=1)
        
        # Plot the path of the second pendulum
        ax3.plot(cart_sol[2, :], cart_sol[3, :], 'b-', alpha=0.5, linewidth=1)
        
        # Highlight current position
        if time_point < cart_sol.shape[1]:
            # Current arms as bold lines
            ax3.plot([0, cart_sol[0, time_point]], [0, cart_sol[1, time_point]], 'r-', linewidth=2)
            ax3.plot([cart_sol[0, time_point], cart_sol[2, time_point]], 
                     [cart_sol[1, time_point], cart_sol[3, time_point]], 'r-', linewidth=2)
            
            # Current positions as points
            ax3.plot([cart_sol[0, time_point]], [cart_sol[1, time_point]], 'ro', markersize=8)
            ax3.plot([cart_sol[2, time_point]], [cart_sol[3, time_point]], 'ro', markersize=8)
        
        ax3.set_xlabel("x position")
        ax3.set_ylabel("y position")
        ax3.grid(True)
        ax3.set_title("Cartesian Trajectory")
        ax3.set_aspect('equal')  # Equal aspect ratio
        
        # Set limits with some padding
        max_val = np.max(np.abs(cart_sol))
        ax3.set_xlim(-max_val*1.1, max_val*1.1)
        ax3.set_ylim(-max_val*1.1, max_val*1.1)
    
    plt.tight_layout()
    return fig_phase

# Function to plot Poincaré section
def plot_poincare_section(poincare_points, section_axis, section_value, section_direction, system_type=""):
    """
    Plot Poincaré section for various system types with enhanced visualization.
    
    Args:
        poincare_points: Array of intersection points
        section_axis: Axis/index used for the section
        section_value: Value at which the section was taken
        section_direction: Direction of crossing
        system_type: Type of dynamical system
        
    Returns:
        Tuple of (figure, number of points)
    """
    if len(poincare_points) == 0:
        return None, 0
    
    # Convert axis index to a string name if it's a number
    axis_name = section_axis
    if isinstance(section_axis, int):
        # Map index to variable name based on system type
        if system_type == "Harmonic Oscillator":
            axis_map = {0: "Position (x)", 1: "Velocity (v)"}
        elif system_type in ["Sinusoidal Function", "Linear Function"]:
            axis_map = {0: "Value (y)", 1: "Derivative (dy/dt)"}
        elif system_type == "Lotka-Volterra":
            axis_map = {0: "Prey Population (x)", 1: "Predator Population (y)"}
        elif system_type == "FitzHugh-Nagumo":
            axis_map = {0: "Membrane Potential (v)", 1: "Recovery Variable (w)"}
        elif system_type == "Coupled Linear System":
            axis_map = {0: "x", 1: "y"}
        elif system_type == "Van der Pol Oscillator" or system_type == "Duffing Oscillator":
            axis_map = {0: "Position (x)", 1: "Velocity (y)"}
        elif system_type == "Double Pendulum":
            axis_map = {0: "Angle 1 (θ₁)", 1: "Angular Velocity 1 (ω₁)", 
                        2: "Angle 2 (θ₂)", 3: "Angular Velocity 2 (ω₂)"}
        elif system_type == "Lorenz System":
            axis_map = {0: "x", 1: "y", 2: "z"}
        else:
            axis_map = {0: "x", 1: "y", 2: "z", 3: "w"}
            
        # Get axis name from map or use generic name
        axis_name = axis_map.get(section_axis, f"Variable {section_axis}")
    
    # Create figure
    fig_poincare = plt.figure(figsize=(10, 8))
    
    # Direction string for the title
    direction_str = "Positive" if section_direction > 0 else "Negative" if section_direction < 0 else "Both"
    
    # Create a custom colormap with better contrast
    cm = plt.cm.viridis.copy()
    
    # Handle based on the dimension of poincare_points
    if poincare_points.ndim == 2 and poincare_points.shape[1] == 2:
        # 3D systems (resulting in 2D sections) or double pendulum's special case
        ax_poincare = fig_poincare.add_subplot(111)
        
        # Create color sequence based on point order
        colors = np.linspace(0, 1, len(poincare_points))
        
        # Use scatter with colormap
        scatter = ax_poincare.scatter(
            poincare_points[:, 0], 
            poincare_points[:, 1], 
            s=50,  # Larger markers
            alpha=0.7, 
            c=colors, 
            cmap=cm,
            edgecolors='w',  # White edges help distinguish points
            linewidths=0.5
        )
        
        # Try to determine labels based on system type and section axis
        if system_type == "Double Pendulum":
            if section_axis == 0:  # Sectioning by θ₁
                x_label = "θ₂"
                y_label = "ω₂"
            elif section_axis == 2:  # Sectioning by θ₂
                x_label = "θ₁"
                y_label = "ω₁"
            elif section_axis == 1:  # Sectioning by ω₁
                x_label = "θ₂"
                y_label = "ω₂"
            elif section_axis == 3:  # Sectioning by ω₂
                x_label = "θ₁"
                y_label = "ω₁"
            else:
                x_label = "Variable 1"
                y_label = "Variable 2"
        elif system_type == "Lorenz System":
            if section_axis == 0:  # Sectioning by x
                x_label = "y"
                y_label = "z"
            elif section_axis == 1:  # Sectioning by y
                x_label = "x"
                y_label = "z"
            elif section_axis == 2:  # Sectioning by z
                x_label = "x"
                y_label = "y"
            else:
                x_label = "Variable 1"
                y_label = "Variable 2"
        else:
            x_label = "Variable 1"
            y_label = "Variable 2"
        
        ax_poincare.set_xlabel(x_label, fontsize=12)
        ax_poincare.set_ylabel(y_label, fontsize=12)
        
        # Add colorbar for temporal sequence
        cbar = plt.colorbar(scatter, ax=ax_poincare)
        cbar.set_label('Sequence', fontsize=12)
        
        # Add connecting lines to show the sequence
        # Only if not too many points to avoid visual clutter
        if len(poincare_points) < 500:
            ax_poincare.plot(
                poincare_points[:, 0], 
                poincare_points[:, 1], 
                'k-', 
                alpha=0.3, 
                linewidth=0.5
            )
        
        # For Lorenz system with z section, add annotations about the lobes
        if system_type == "Lorenz System" and section_axis == 2 and abs(section_value - 27) < 3:
            # Find centers of the two lobes (this is an approximation)
            if len(poincare_points) > 10:
                from scipy.cluster.vq import kmeans
                centers, _ = kmeans(poincare_points, 2)
                
                for i, center in enumerate(centers):
                    ax_poincare.annotate(
                        f"Lobe {i+1}",
                        xy=(center[0], center[1]),
                        xytext=(center[0] + 3, center[1] + 3),
                        fontsize=10,
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
                    )
        
    elif poincare_points.ndim == 1 or (poincare_points.ndim == 2 and poincare_points.shape[1] == 1):
        # 2D systems (resulting in 1D sections)
        # We'll create two plots: a time series and a return map
        
        # Plot 1: Time series of intersection values
        ax1 = fig_poincare.add_subplot(211)
        
        # Ensure points is a 1D array
        if poincare_points.ndim == 2:
            points = poincare_points.flatten()
        else:
            points = poincare_points
            
        iterations = np.arange(len(points))
        
        # Create color sequence based on point order
        colors = np.linspace(0, 1, len(points))
        
        # Scatter plot with colored sequence
        scatter1 = ax1.scatter(
            iterations, 
            points, 
            s=50,  # Larger markers
            alpha=0.7, 
            c=colors, 
            cmap=cm,
            edgecolors='w',  # White edges
            linewidths=0.5
        )
        
        # Line connecting the points
        ax1.plot(iterations, points, 'k-', alpha=0.5, linewidth=0.8)
        
        # Set labels
        ax1.set_xlabel('Crossing Number', fontsize=12)
        
        # Determine which variable is being plotted (not the sectioning variable)
        if system_type == "Harmonic Oscillator":
            y_var_name = "Velocity (v)" if section_axis == 0 else "Position (x)"
        elif system_type in ["Sinusoidal Function", "Linear Function"]:
            y_var_name = "Derivative (dy/dt)" if section_axis == 0 else "Value (y)"
        elif system_type == "Lotka-Volterra":
            y_var_name = "Predator Population (y)" if section_axis == 0 else "Prey Population (x)"
        elif system_type == "FitzHugh-Nagumo":
            y_var_name = "Recovery Variable (w)" if section_axis == 0 else "Membrane Potential (v)"
        elif system_type == "Coupled Linear System":
            y_var_name = "y" if section_axis == 0 else "x"
        elif system_type in ["Van der Pol Oscillator", "Duffing Oscillator"]:
            y_var_name = "Velocity (y)" if section_axis == 0 else "Position (x)"
        else:
            y_var_name = "Other Variable"
            
        ax1.set_ylabel(f'{y_var_name} at crossings', fontsize=12)
        ax1.set_title(f'Poincaré Section Values (Direction: {direction_str})', fontsize=14)
        ax1.grid(True)
        
        # Add annotations for important features
        if len(points) > 3:
            # Find maximum and minimum points
            max_idx = np.argmax(points)
            min_idx = np.argmin(points)
            
            # Annotate max and min
            ax1.annotate(
                f"Max: {points[max_idx]:.3f}",
                xy=(iterations[max_idx], points[max_idx]),
                xytext=(iterations[max_idx], points[max_idx] + (np.max(points) - np.min(points))/10),
                fontsize=10,
                arrowprops=dict(facecolor='red', shrink=0.05),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
            
            ax1.annotate(
                f"Min: {points[min_idx]:.3f}",
                xy=(iterations[min_idx], points[min_idx]),
                xytext=(iterations[min_idx], points[min_idx] - (np.max(points) - np.min(points))/10),
                fontsize=10,
                arrowprops=dict(facecolor='blue', shrink=0.05),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
            
            # Add colorbar for point sequence
            plt.colorbar(scatter1, ax=ax1, pad=0.01, label='Sequence')
        
        # Plot 2: Return map (x_n vs x_n+1)
        if len(points) > 3:  # Need at least a few points for a return map
            ax2 = fig_poincare.add_subplot(212)
            
            # Create scatter plot with colored sequence
            scatter2 = ax2.scatter(
                points[:-1], 
                points[1:], 
                s=50,  # Larger markers
                alpha=0.7, 
                c=colors[:-1], 
                cmap=cm, 
                edgecolors='w',
                linewidths=0.5
            )
            
            # Add connecting lines if not too many points
            if len(points) < 100:
                ax2.plot(points[:-1], points[1:], 'k-', alpha=0.3, linewidth=0.5)
            
            # Add the identity line x_n+1 = x_n
            min_val = min(points)
            max_val = max(points)
            padding = 0.1 * (max_val - min_val)
            line_range = [min_val - padding, max_val + padding]
            ax2.plot(line_range, line_range, 'r--', alpha=0.5, linewidth=1)
            
            ax2.set_xlabel(f'{y_var_name}_n', fontsize=12)
            ax2.set_ylabel(f'{y_var_name}_{{n+1}}', fontsize=12)
            ax2.set_title('Return Map', fontsize=14)
            ax2.grid(True)
            
            # Keep axes square and matching limits
            ax2.set_aspect('equal')
            ax2.set_xlim(line_range)
            ax2.set_ylim(line_range)
            
            # Add colorbar for the sequence
            plt.colorbar(scatter2, ax=ax2, pad=0.01, label='Sequence')
            
            # Add analysis of return map
            if system_type in ["Harmonic Oscillator", "Linear Function"]:
                # Look for periodic behavior in return map
                if len(points) > 10:
                    # Calculate distances between consecutive points
                    diffs = np.abs(points[1:] - points[:-1])
                    avg_diff = np.mean(diffs)
                    std_diff = np.std(diffs)
                    
                    if std_diff < avg_diff * 0.2:  # Low variation indicates periodicity
                        period_est = 1  # Default period estimate
                        
                        # Try to estimate period by looking for repeating patterns
                        if len(points) > 20:
                            # Use autocorrelation to estimate periodicity
                            from scipy import signal
                            autocorr = signal.correlate(points, points, mode='full')
                            autocorr = autocorr[len(autocorr)//2:]
                            
                            # Find peaks in autocorrelation (after the first one at lag 0)
                            peaks, _ = signal.find_peaks(autocorr)
                            if len(peaks) > 0:
                                period_est = peaks[0]
                                
                                # Add annotation to the plot
                                if period_est > 1:
                                    ax2.annotate(
                                        f"Period ≈ {period_est}",
                                        xy=(0.05, 0.95),
                                        xycoords='axes fraction',
                                        fontsize=12,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
                                    )
    else:
        # Handle other cases or custom systems
        ax_poincare = fig_poincare.add_subplot(111)
        ax_poincare.text(0.5, 0.5, "Custom Poincaré section visualization not implemented\nfor this system type and dimension", 
                        ha='center', va='center', transform=ax_poincare.transAxes)
    
    # Add system info in the figure suptitle
    title = f"Poincaré Section for {system_type} at {axis_name}={section_value:.4f}"
    plt.suptitle(title, fontsize=14)
        
    plt.tight_layout()
    if poincare_points.ndim == 2 and poincare_points.shape[1] == 2:
        plt.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    return fig_poincare, len(poincare_points)

# Function to plot top features bar chart
def plot_top_features(latent_features, time_point, feature_idx, top_n=50):
    """Create bar chart of top feature activations at a time point."""
    features_at_time = latent_features[time_point, :]
    sorted_indices = np.argsort(features_at_time)[::-1]
    top_indices = sorted_indices[:top_n]
    
    # Create bar chart with gradient coloring
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 4))
    
    # Create a colormap for the bars
    bar_colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
    
    # Create bars with thin spacing
    bars = ax3.bar(top_indices, features_at_time[top_indices], 
                  color=bar_colors, width=0.8, 
                  edgecolor='none', alpha=0.8)
    
    # Highlight the currently selected feature if it's in the top N
    if feature_idx in top_indices:
        feature_pos = np.where(top_indices == feature_idx)[0][0]
        bars[feature_pos].set_color('red')
        bars[feature_pos].set_edgecolor('black')
        bars[feature_pos].set_linewidth(1.5)
        highlight_info = f"Feature {feature_idx} is highlighted in red"
    else:
        highlight_info = f"Feature {feature_idx} is not among the top {top_n} features at this time point"
        
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Activation Value')
    ax3.set_title(f'Top {top_n} Feature Activations at Time Point {time_point}')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Use a cleaner background
    ax3.set_facecolor('#f8f9fa')
    fig3.patch.set_facecolor('#f8f9fa')
    
    return fig3, highlight_info

# Function to plot all features heatmap
def plot_features_heatmap(latent_features, time_point, feature_idx, highlight_top_n=True, top_n_global=100):
    """Plot heatmap of all features over time with optional highlighting."""
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 4))
    
    # Get the base heatmap
    heatmap_data = latent_features.T
    hm = ax4.imshow(heatmap_data, aspect='auto', cmap='viridis')
    
    # Find top N activations globally if highlighting
    top_activations_data = []
    if highlight_top_n:
        flattened = latent_features.flatten()
        threshold = np.sort(flattened)[-min(top_n_global, len(flattened))]
        
        # Find time points with activations over threshold and their values
        rows, cols = np.where(heatmap_data >= threshold)
        values = heatmap_data[rows, cols]
        
        # Plot circles at top activation locations
        scatter = ax4.scatter(cols, rows, s=15, c='red', alpha=0.5, 
                            marker='o', edgecolors='white', linewidth=0.5)
        
        # Count top activations per feature
        feature_counts = {}
        for r in rows:
            if r not in feature_counts:
                feature_counts[r] = 0
            feature_counts[r] += 1
        
        current_feature_count = feature_counts.get(feature_idx, 0)
        
        # Prepare data for table
        for i in range(len(values)):
            f_idx = int(rows[i])
            t_idx = int(cols[i])
            
            data_entry = {
                "Feature": f_idx,
                "Time Point": t_idx,
                "Activation": f"{values[i]:.4f}"
            }
            top_activations_data.append(data_entry)
            
        # Sort by activation value (descending)
        top_activations_data.sort(key=lambda x: float(x["Activation"]), reverse=True)
    
    # Common plot elements
    ax4.set_xlabel('Time Point')
    ax4.set_ylabel('Feature Index')
    ax4.set_title('All Features Over Time')
    
    # Mark selected time point
    ax4.axvline(x=time_point, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
    
    # Mark selected feature
    ax4.axhline(y=feature_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        
    plt.colorbar(hm, ax=ax4, label='Activation Value')
    
    return fig4, top_activations_data, threshold if highlight_top_n else None

# Function for dimensionality reduction visualization
def compute_dimensionality_reduction(data, method, n_components, **kwargs):
    """
    Compute dimensionality reduction with caching.
    
    Args:
        data: Input data (n_samples, n_features)
        method: 'tsne' or 'umap'
        n_components: 2 or 3
        **kwargs: Additional parameters for the method
        
    Returns:
        Reduced dimensionality data
    """
    try:
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                **kwargs
            )
        elif method == 'umap':
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return reducer.fit_transform(data)
    except ImportError:
        st.error(f"Required libraries for {method} not installed.")
        return None