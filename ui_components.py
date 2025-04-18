import streamlit as st
import numpy as np
from pathlib import Path
import pandas as pd
import importlib

def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.header("Setup")
    model_dir = st.sidebar.text_input("Model directory", "./odeformer/")
    sae_dir = st.sidebar.text_input("SAE directory", "./sae")

    # Add the paths
    paths_added = False
    if st.sidebar.button("Add Paths") or 'paths_added' not in st.session_state:
        import sys
        if model_dir and Path(model_dir).exists():
            if model_dir not in sys.path:
                sys.path.append(model_dir)
            if f"{model_dir}/odeformer" not in sys.path:
                sys.path.append(f"{model_dir}/odeformer")
            if f"{model_dir}/odeformer/model" not in sys.path:
                sys.path.append(f"{model_dir}/odeformer/model")
            if f"{model_dir}/odeformer/envs" not in sys.path:
                sys.path.append(f"{model_dir}/odeformer/envs")
            st.sidebar.success(f"Added {model_dir} to sys.path")
            paths_added = True
        else:
            st.sidebar.error(f"Directory {model_dir} not found")
            
        if sae_dir and Path(sae_dir).exists():
            if sae_dir not in sys.path:
                sys.path.append(sae_dir)
            st.sidebar.success(f"Added {sae_dir} to sys.path")
            paths_added = True
        else:
            st.sidebar.error(f"Directory {sae_dir} not found")
            
        # Update session state
        st.session_state.paths_added = paths_added
        
    # Check if required libraries are available
    check_libraries()
    
    return st.session_state.paths_added

def check_libraries():
    """Check for required libraries and provide warnings/instructions if missing."""
    # Check for mishax for instrumentation
    mishax_available = False
    try:
        from mishax import ast_patcher, safe_greenlet
        mishax_available = True
        st.sidebar.success("✅ Found mishax libraries - instrumentation available")
    except ImportError:
        st.sidebar.warning("⚠️ mishax libraries not found - using synthetic activations")
        st.sidebar.markdown("""
        To enable full instrumentation, install mishax:
        ```
        pip install mishax
        ```
        """)

    # Check for dimensionality reduction libraries
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    umap_available = importlib.util.find_spec("umap") is not None
    
    if sklearn_available and umap_available:
        st.sidebar.success("✅ Found dimensionality reduction libraries")
    else:
        missing_libs = []
        if not sklearn_available:
            missing_libs.append("scikit-learn")
        if not umap_available:
            missing_libs.append("umap-learn")
            
        missing_libs_str = ", ".join(missing_libs)
        st.sidebar.warning(f"⚠️ Missing libraries for dimensionality reduction: {missing_libs_str}")
        st.sidebar.markdown(f"""
        To enable t-SNE and UMAP visualizations, install:
        ```
        pip install {' '.join(missing_libs)}
        ```
        """)
    
    return {
        'mishax': mishax_available,
        'sklearn': sklearn_available,
        'umap': umap_available
    }

def setup_activation_ui():
    """Setup UI for selecting activation site and component."""
    if not st.session_state.models_loaded:
        return
        
    st.sidebar.header("Activation Settings")
    
    # If we've collected activations before, get the available sites
    available_sites = []
    if st.session_state.all_collected_activations:
        # Convert site keys to strings to ensure compatibility
        available_sites = [str(site) for site in st.session_state.all_collected_activations.keys()]
    
    # If no activations collected yet, use only the known available sites
    if not available_sites:
        available_sites = ['RESIDUAL', 'ATTN_OUTPUT']
    
    # Site selection dropdown
    site_index = 0
    if st.session_state.activation_site in available_sites:
        site_index = available_sites.index(st.session_state.activation_site)
        
    selected_site = st.sidebar.selectbox(
        "Activation Site", 
        available_sites,
        index=site_index
    )
    
    # Component selection dropdown - only show components that exist for this site
    available_components = []
    
    if st.session_state.all_collected_activations:
        # Find the matching site key - handle both string and enum cases
        matching_site_key = None
        for site_key in st.session_state.all_collected_activations.keys():
            if str(site_key) == selected_site:
                matching_site_key = site_key
                break
        
        if matching_site_key is not None:
            # Get available components for this site
            available_components = list(st.session_state.all_collected_activations[matching_site_key].keys())
    
    # If no components are available for this site, show an empty component dropdown
    component_index = 0
    component_disabled = False
    
    if not available_components:
        available_components = ["No components available for this site"]
        component_disabled = True
    elif st.session_state.activation_component in available_components:
        component_index = available_components.index(st.session_state.activation_component)
        
    selected_component = st.sidebar.selectbox(
        "Component", 
        available_components,
        index=component_index,
        disabled=component_disabled
    )
    
    # Only enable custom component input if we're not showing a placeholder
    use_custom = False
    if not component_disabled:
        use_custom = st.sidebar.checkbox("Use custom component", False)
    
    if use_custom:
        custom_component = st.sidebar.text_input("Custom Component", st.session_state.activation_component)
        selected_component = custom_component
    
    # Check if selection changed (don't update if we're in the disabled state)
    if (not component_disabled and
        (selected_site != st.session_state.activation_site or 
        selected_component != st.session_state.activation_component)):
        # Update session state
        st.session_state.activation_site = selected_site
        st.session_state.activation_component = selected_component
        # Clear cached activations and features
        st.session_state.current_activations = None
        st.session_state.current_latent_features = None
        st.sidebar.info("Activation selection changed - recollecting activations")

def setup_system_selection_ui():
    """Setup UI for selecting system type and parameters."""
    from session_state import get_all_system_types
    
    st.sidebar.header("System Selection")
    system_types = get_all_system_types()
    system_type = st.sidebar.selectbox(
        "Select System Type",
        system_types,
        index=system_types.index(st.session_state.system_type) if st.session_state.system_type in system_types else 0
    )
    st.session_state.system_type = system_type

    # Initial conditions and time span
    st.sidebar.header("Time Settings")
    t_start = st.sidebar.number_input("Start Time", value=0.0, step=1.0)
    
    # Adjust end time based on system type
    if system_type == "FitzHugh-Nagumo":
        t_end_default = 100.0
    elif system_type in ["Lotka-Volterra", "Van der Pol Oscillator", "Double Pendulum"]:
        t_end_default = 20.0
    elif system_type in ["Duffing Oscillator", "Lorenz System"]:
        t_end_default = 50.0
    elif system_type in ["Sigmoid Function", "Tanh Function"]:
        t_end_default = 10.0
        t_start = -5.0  # Better range for sigmoid/tanh
    else:
        t_end_default = 10.0
        
    t_end = st.sidebar.number_input("End Time", value=t_end_default, step=1.0, min_value=t_start + 0.1)
    
    # Adjust number of points based on system type
    if system_type in ["Duffing Oscillator", "Lorenz System"]:
        t_points_default = 500
    elif system_type == "FitzHugh-Nagumo":
        t_points_default = 1000
    else:
        t_points_default = 200
        
    t_points = st.sidebar.slider("Number of Time Points", 10, 5000, t_points_default, 10)
    times = np.linspace(t_start, t_end, t_points)
    
    # Parameter inputs based on system selection
    st.sidebar.header("System Parameters")
    
    # Collect parameters based on system type
    params = setup_system_params_ui(system_type)
    
    return system_type, times, params

def setup_system_params_ui(system_type):
    """Setup UI for system-specific parameters."""
    if system_type == "Harmonic Oscillator":
        return setup_harmonic_oscillator_ui()
    elif system_type == "Sinusoidal Function":
        return setup_sinusoidal_ui()
    elif system_type == "Linear Function":
        return setup_linear_ui()
    elif system_type == "Simple Exponential":
        return setup_exponential_ui()
    elif system_type == "Simple Polynomial":
        return setup_polynomial_ui()
    elif system_type == "Sigmoid Function":
        return setup_sigmoid_ui()
    elif system_type == "Tanh Function":
        return setup_tanh_ui()
    elif system_type == "Lotka-Volterra":
        return setup_lotka_volterra_ui()
    elif system_type == "FitzHugh-Nagumo":
        return setup_fitzhugh_nagumo_ui()
    elif system_type == "Coupled Linear System":
        return setup_coupled_linear_ui()
    elif system_type == "Van der Pol Oscillator":
        return setup_van_der_pol_ui()
    elif system_type == "Duffing Oscillator":
        return setup_duffing_ui()
    elif system_type == "Double Pendulum":
        return setup_double_pendulum_ui()
    elif system_type == "Lorenz System":
        return setup_lorenz_ui()
    else:
        st.sidebar.warning(f"No parameters UI for system type: {system_type}")
        return {}

def setup_poincare_section_ui(system_type, solution=None, key_suffix=""):
    """
    Setup UI for Poincaré section parameters for any system type with enhanced flexibility.
    
    Args:
        system_type: The type of dynamical system
        solution: Optional solution data to help set appropriate parameter ranges
        key_suffix: Optional suffix to make keys unique (prevents duplicate key errors)
    """
    # Create a unique key suffix if none is provided
    if not key_suffix:
        import time
        key_suffix = f"_{int(time.time() * 1000) % 10000}"  # Use current time as unique identifier
    
    st.sidebar.subheader("Poincaré Section")
    
    # Map system variables to display names and indices
    variable_map = {}
    
    if system_type == "Harmonic Oscillator":
        variable_map = {
            "Position (x)": {"index": 0, "name": "x", "default_value": 0.0},
            "Velocity (v)": {"index": 1, "name": "y", "default_value": 0.0}
        }
    elif system_type in ["Sinusoidal Function", "Linear Function", "Simple Exponential", 
                         "Simple Polynomial", "Sigmoid Function", "Tanh Function"]:
        variable_map = {
            "Value (y)": {"index": 0, "name": "y", "default_value": 0.0},
            "Derivative (dy/dt)": {"index": 1, "name": "dy/dt", "default_value": 0.0}
        }
    elif system_type == "Lotka-Volterra":
        variable_map = {
            "Prey Population (x)": {"index": 0, "name": "x", "default_value": 1.0},
            "Predator Population (y)": {"index": 1, "name": "y", "default_value": 0.5}
        }
    elif system_type == "FitzHugh-Nagumo":
        variable_map = {
            "Membrane Potential (v)": {"index": 0, "name": "v", "default_value": 0.0},
            "Recovery Variable (w)": {"index": 1, "name": "w", "default_value": 0.0}
        }
    elif system_type == "Coupled Linear System":
        variable_map = {
            "x": {"index": 0, "name": "x", "default_value": 0.0},
            "y": {"index": 1, "name": "y", "default_value": 0.0}
        }
    elif system_type == "Van der Pol Oscillator":
        variable_map = {
            "Position (x)": {"index": 0, "name": "x", "default_value": 0.0},
            "Velocity (y)": {"index": 1, "name": "y", "default_value": 0.0}
        }
    elif system_type == "Duffing Oscillator":
        variable_map = {
            "Position (x)": {"index": 0, "name": "x", "default_value": 0.0},
            "Velocity (y)": {"index": 1, "name": "y", "default_value": 0.0}
        }
    elif system_type == "Double Pendulum":
        variable_map = {
            "Angle 1 (θ₁)": {"index": 0, "name": "θ₁", "default_value": 0.0},
            "Angular Velocity 1 (ω₁)": {"index": 1, "name": "ω₁", "default_value": 0.0},
            "Angle 2 (θ₂)": {"index": 2, "name": "θ₂", "default_value": 0.0},
            "Angular Velocity 2 (ω₂)": {"index": 3, "name": "ω₂", "default_value": 0.0}
        }
    elif system_type == "Lorenz System":
        variable_map = {
            "x": {"index": 0, "name": "x", "default_value": 0.0},
            "y": {"index": 1, "name": "y", "default_value": 0.0},
            "z": {"index": 2, "name": "z", "default_value": 27.0}  # Classic value for Lorenz
        }
    
    # Variable names for display in selectbox
    variable_names = list(variable_map.keys())

    # If we have the solution data, calculate appropriate min/max values for each variable
    if solution is not None and 'solution' in solution:
        solution_data = solution['solution']
        for i, var_name in enumerate(variable_names):
            if i < solution_data.shape[1]:  # Make sure index is valid
                var_min = float(solution_data[:, i].min())
                var_max = float(solution_data[:, i].max())
                # Add to the variable map
                variable_map[var_name]["min"] = var_min
                variable_map[var_name]["max"] = var_max
                # Set a more reasonable default value based on the data range
                variable_map[var_name]["default_value"] = (var_min + var_max) / 2
    
    # Get or initialize the current axis
    current_variable = st.session_state.poincare_params.get('variable', variable_names[0])
    if current_variable not in variable_names:
        current_variable = variable_names[0]
    
    # Map the user-friendly variable name to its index and internal name
    current_index = variable_map[current_variable]["index"]
    
    # Allow user to select variable by name - USE UNIQUE KEY
    selected_variable = st.sidebar.selectbox(
        "Section Variable", 
        variable_names,
        index=variable_names.index(current_variable),
        key=f"poincare_variable_{system_type}{key_suffix}"  # Add suffix to make key unique
    )
    
    # Update session state with selected variable info
    selected_var_info = variable_map[selected_variable]
    st.session_state.poincare_params['variable'] = selected_variable
    st.session_state.poincare_params['axis'] = selected_var_info["index"]
    
    # Set min/max values for the slider based on the solution data or defaults
    if "min" in selected_var_info and "max" in selected_var_info:
        value_min = selected_var_info["min"]
        value_max = selected_var_info["max"]
        # Add a bit of padding to the range
        range_padding = (value_max - value_min) * 0.1
        value_min -= range_padding
        value_max += range_padding
    else:
        # Default ranges if no solution data available
        if system_type == "Double Pendulum" and selected_variable in ["Angle 1 (θ₁)", "Angle 2 (θ₂)"]:
            value_min = -np.pi
            value_max = np.pi
        elif system_type == "Lorenz System" and selected_variable == "z":
            value_min = 0.0
            value_max = 50.0
        else:
            value_min = -10.0
            value_max = 10.0
    
    # Get default value from the variable info
    value_default = selected_var_info.get("default_value", 0.0)
    
    # Use the current value from session state if available, otherwise use default
    current_value = st.session_state.poincare_params.get('value', value_default)
    
    # Ensure current value is within the slider range
    current_value = min(max(current_value, value_min), value_max)
    
    # Create step size based on range (for better slider usability)
    step_size = (value_max - value_min) / 100.0
    step_size = max(0.001, min(step_size, 0.1))  # Keep between 0.001 and 0.1
    
    # Section Value slider - USE UNIQUE KEY
    value = st.sidebar.slider(
        "Section Value", 
        float(value_min), 
        float(value_max), 
        float(current_value), 
        step=float(step_size),
        key=f"poincare_value_slider_{system_type}{key_suffix}",  # Add suffix to key
        help="Value at which to take the section"
    )
    st.session_state.poincare_params['value'] = value
    
    # Add a more precise number input for fine control - USE UNIQUE KEY
    value = st.sidebar.number_input(
        "Precise Value", 
        value_min, 
        value_max, 
        st.session_state.poincare_params['value'], 
        step=step_size/10,  # Finer step for precise control
        key=f"poincare_value_input_{system_type}{key_suffix}",  # Add suffix to key
        help="Precise value for the section"
    )
    st.session_state.poincare_params['value'] = value
    
    # Direction options
    direction_options = [(1, "Positive"), (-1, "Negative"), (0, "Both")]
    
    # Get current direction or default to 1
    current_direction = st.session_state.poincare_params.get('direction', 1)
    direction_index = 0
    
    # Find the index of the current direction in the options
    for i, (d, _) in enumerate(direction_options):
        if d == current_direction:
            direction_index = i
            break
    
    # Direction selectbox - USE UNIQUE KEY
    selected_direction = st.sidebar.selectbox(
        "Crossing Direction", 
        direction_options,
        index=direction_index,
        format_func=lambda x: x[1],
        key=f"poincare_direction_{system_type}{key_suffix}",  # Add suffix to key
        help="Direction in which variable crosses the section value"
    )
    
    st.session_state.poincare_params['direction'] = selected_direction[0]
    
    # Add explanation for specific system types
    if system_type == "Harmonic Oscillator":
        if selected_variable == "Position (x)" and abs(value) < 0.1:
            st.sidebar.info("Setting position near zero shows velocity when the oscillator crosses the origin.")
    elif system_type == "Van der Pol Oscillator":
        if selected_variable == "Position (x)" and abs(value) < 0.1:
            st.sidebar.info("Setting position near zero reveals the limit cycle structure.")
    elif system_type == "Lorenz System":
        if selected_variable == "z" and abs(value - 27) < 3:
            st.sidebar.info("z ≈ 27 (rho parameter) shows the classic butterfly pattern.")
    
    # Return the section parameters for convenience
    return {
        'variable': selected_variable,
        'axis_index': st.session_state.poincare_params['axis'],
        'value': value,
        'direction': selected_direction[0]
    }

def setup_harmonic_oscillator_ui():
    """Setup UI for harmonic oscillator system."""
    st.sidebar.subheader("Equation Parameters")
    # Use columns for sliders and text inputs
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        omega = st.slider("Natural Frequency (ω)", 0.1, 50.0, st.session_state.get('ho_omega', 1.0), 0.1)
        st.session_state.ho_omega = omega
        
    with col2:
        omega = st.number_input("ω value", 0.1, 50.0, st.session_state.ho_omega, 0.1, key="ho_omega_input")
        st.session_state.ho_omega = omega
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        gamma = st.slider("Damping Coefficient (γ)", 0.0, 50.0, st.session_state.get('ho_gamma', 0.5), 0.1)
        st.session_state.ho_gamma = gamma
        
    with col2:
        gamma = st.number_input("γ value", 0.0, 50.0, st.session_state.ho_gamma, 0.1, key="ho_gamma_input")
        st.session_state.ho_gamma = gamma
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.get('ho_x0', 1.0), 0.1)
        st.session_state.ho_x0 = x0
        
    with col2:
        x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.ho_x0, 0.1, key="ho_x0_input")
        st.session_state.ho_x0 = x0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        v0 = st.slider("Initial Velocity (v₀)", -50.0, 50.0, st.session_state.get('ho_v0', 0.0), 0.1)
        st.session_state.ho_v0 = v0
        
    with col2:
        v0 = st.number_input("v₀ value", -50.0, 50.0, st.session_state.ho_v0, 0.1, key="ho_v0_input")
        st.session_state.ho_v0 = v0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Harmonic Oscillator")
    
    return (omega, gamma, x0, v0)

def setup_sinusoidal_ui():
    """Setup UI for sinusoidal function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        amplitude = st.slider("Amplitude (A)", 0.1, 50.0, st.session_state.get('sin_amplitude', 1.0), 0.1)
        st.session_state.sin_amplitude = amplitude
        
    with col2:
        amplitude = st.number_input("A value", 0.1, 50.0, st.session_state.sin_amplitude, 0.1, key="sin_amp_input")
        st.session_state.sin_amplitude = amplitude
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        frequency = st.slider("Frequency (ω)", 0.1, 50.0, st.session_state.get('sin_frequency', 1.0), 0.1)
        st.session_state.sin_frequency = frequency
        
    with col2:
        frequency = st.number_input("ω value", 0.1, 50.0, st.session_state.sin_frequency, 0.1, key="sin_freq_input")
        st.session_state.sin_frequency = frequency
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        phase = st.slider("Phase (φ, radians)", 0.0, 2*np.pi, st.session_state.get('sin_phase', 0.0), 0.1)
        st.session_state.sin_phase = phase
        
    with col2:
        phase = st.number_input("φ value", 0.0, 2*np.pi, st.session_state.sin_phase, 0.1, key="sin_phase_input")
        st.session_state.sin_phase = phase
        
    use_cos = st.sidebar.checkbox("Use Cosine instead of Sine", st.session_state.get('sin_use_cos', False))
    st.session_state.sin_use_cos = use_cos
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Sinusoidal Function")
    
    return (amplitude, frequency, phase, use_cos)

def setup_linear_ui():
    """Setup UI for linear function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        slope = st.slider("Slope (m)", -50.0, 50.0, st.session_state.get('lin_slope', 1.0), 0.1)
        st.session_state.lin_slope = slope
        
    with col2:
        slope = st.number_input("m value", -50.0, 50.0, st.session_state.lin_slope, 0.1, key="lin_slope_input")
        st.session_state.lin_slope = slope
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        intercept = st.slider("Intercept (b)", -50.0, 50.0, st.session_state.get('lin_intercept', 0.0), 0.1)
        st.session_state.lin_intercept = intercept
        
    with col2:
        intercept = st.number_input("b value", -50.0, 50.0, st.session_state.lin_intercept, 0.1, key="lin_intercept_input")
        st.session_state.lin_intercept = intercept
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Linear Function")
    
    return (slope, intercept)

def setup_lotka_volterra_ui():
    """Setup UI for Lotka-Volterra system."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        alpha = st.slider("Prey Growth Rate (α)", 0.1, 20.0, st.session_state.get('lv_alpha', 0.5), 0.1)
        st.session_state.lv_alpha = alpha
        
    with col2:
        alpha = st.number_input("α value", 0.1, 20.0, st.session_state.lv_alpha, 0.1, key="lv_alpha_input")
        st.session_state.lv_alpha = alpha
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Predation Rate (β)", 0.01, 10.0, st.session_state.get('lv_beta', 0.2), 0.01)
        st.session_state.lv_beta = beta
        
    with col2:
        beta = st.number_input("β value", 0.01, 10.0, st.session_state.lv_beta, 0.01, key="lv_beta_input")
        st.session_state.lv_beta = beta
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        delta = st.slider("Predator Death Rate (δ)", 0.1, 20.0, st.session_state.get('lv_delta', 0.5), 0.1)
        st.session_state.lv_delta = delta
        
    with col2:
        delta = st.number_input("δ value", 0.1, 20.0, st.session_state.lv_delta, 0.1, key="lv_delta_input")
        st.session_state.lv_delta = delta
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        gamma = st.slider("Predator Growth from Prey (γ)", 0.01, 10.0, st.session_state.get('lv_gamma', 0.1), 0.01)
        st.session_state.lv_gamma = gamma
        
    with col2:
        gamma = st.number_input("γ value", 0.01, 10.0, st.session_state.lv_gamma, 0.01, key="lv_gamma_input")
        st.session_state.lv_gamma = gamma
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        prey0 = st.slider("Initial Prey Population (x₀)", 0.1, 50.0, st.session_state.get('lv_prey0', 1.0), 0.1)
        st.session_state.lv_prey0 = prey0
        
    with col2:
        prey0 = st.number_input("x₀ value", 0.1, 50.0, st.session_state.lv_prey0, 0.1, key="lv_prey0_input")
        st.session_state.lv_prey0 = prey0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        predator0 = st.slider("Initial Predator Population (y₀)", 0.1, 50.0, st.session_state.get('lv_predator0', 0.5), 0.1)
        st.session_state.lv_predator0 = predator0
        
    with col2:
        predator0 = st.number_input("y₀ value", 0.1, 50.0, st.session_state.lv_predator0, 0.1, key="lv_predator0_input")
        st.session_state.lv_predator0 = predator0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Lotka-Volterra")
    
    return (alpha, beta, delta, gamma, prey0, predator0)

def setup_fitzhugh_nagumo_ui():
    """Setup UI for FitzHugh-Nagumo system."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Parameter a", -10.0, 10.0, st.session_state.get('fn_a', 0.7), 0.1)
        st.session_state.fn_a = a
        
    with col2:
        a = st.number_input("a value", -10.0, 10.0, st.session_state.fn_a, 0.1, key="fn_a_input")
        st.session_state.fn_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Parameter b", 0.1, 10.0, st.session_state.get('fn_b', 0.8), 0.1)
        st.session_state.fn_b = b
        
    with col2:
        b = st.number_input("b value", 0.1, 10.0, st.session_state.fn_b, 0.1, key="fn_b_input")
        st.session_state.fn_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tau = st.slider("Time Scale (τ)", 1.0, 200.0, st.session_state.get('fn_tau', 12.5), 0.5)
        st.session_state.fn_tau = tau
        
    with col2:
        tau = st.number_input("τ value", 1.0, 200.0, st.session_state.fn_tau, 0.5, key="fn_tau_input")
        st.session_state.fn_tau = tau
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        I = st.slider("Input Current (I)", 0.0, 20.0, st.session_state.get('fn_I', 0.5), 0.1)
        st.session_state.fn_I = I
        
    with col2:
        I = st.number_input("I value", 0.0, 20.0, st.session_state.fn_I, 0.1, key="fn_I_input")
        st.session_state.fn_I = I
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        v0 = st.slider("Initial Membrane Potential (v₀)", -20.0, 20.0, st.session_state.get('fn_v0', 0.0), 0.1)
        st.session_state.fn_v0 = v0
        
    with col2:
        v0 = st.number_input("v₀ value", -20.0, 20.0, st.session_state.fn_v0, 0.1, key="fn_v0_input")
        st.session_state.fn_v0 = v0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        w0 = st.slider("Initial Recovery Variable (w₀)", -20.0, 20.0, st.session_state.get('fn_w0', 0.0), 0.1)
        st.session_state.fn_w0 = w0
        
    with col2:
        w0 = st.number_input("w₀ value", -20.0, 20.0, st.session_state.fn_w0, 0.1, key="fn_w0_input")
        st.session_state.fn_w0 = w0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("FitzHugh-Nagumo")
    
    return (a, b, tau, I, v0, w0)

def setup_coupled_linear_ui():
    """Setup UI for coupled linear system."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        alpha = st.slider("Alpha (α) coefficient", -50.0, 50.0, st.session_state.get('cl_alpha', 1.0), 0.1)
        st.session_state.cl_alpha = alpha
        
    with col2:
        alpha = st.number_input("α value", -50.0, 50.0, st.session_state.cl_alpha, 0.1, key="cl_alpha_input")
        st.session_state.cl_alpha = alpha
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Beta (β) coefficient", -50.0, 50.0, st.session_state.get('cl_beta', -1.0), 0.1)
        st.session_state.cl_beta = beta
        
    with col2:
        beta = st.number_input("β value", -50.0, 50.0, st.session_state.cl_beta, 0.1, key="cl_beta_input")
        st.session_state.cl_beta = beta
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.slider("Initial x value (x₀)", -50.0, 50.0, st.session_state.get('cl_x0', 1.0), 0.1)
        st.session_state.cl_x0 = x0
        
    with col2:
        x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.cl_x0, 0.1, key="cl_x0_input")
        st.session_state.cl_x0 = x0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        y0 = st.slider("Initial y value (y₀)", -50.0, 50.0, st.session_state.get('cl_y0', 0.0), 0.1)
        st.session_state.cl_y0 = y0
        
    with col2:
        y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.cl_y0, 0.1, key="cl_y0_input")
        st.session_state.cl_y0 = y0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Coupled Linear System")
    
    return (alpha, beta, x0, y0)

def setup_van_der_pol_ui():
    """Setup UI for Van der Pol oscillator."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mu = st.slider("Nonlinearity Parameter (μ)", 0.1, 50.0, st.session_state.van_der_pol_params['mu'], 0.1)
        st.session_state.van_der_pol_params['mu'] = mu
        
    with col2:
        mu = st.number_input("μ value", 0.1, 50.0, st.session_state.van_der_pol_params['mu'], 0.1)
        st.session_state.van_der_pol_params['mu'] = mu
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.van_der_pol_params['x0'], 0.1)
        st.session_state.van_der_pol_params['x0'] = x0
        
    with col2:
        x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.van_der_pol_params['x0'], 0.1)
        st.session_state.van_der_pol_params['x0'] = x0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        y0 = st.slider("Initial Velocity (y₀)", -50.0, 50.0, st.session_state.van_der_pol_params['y0'], 0.1)
        st.session_state.van_der_pol_params['y0'] = y0
        
    with col2:
        y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.van_der_pol_params['y0'], 0.1)
        st.session_state.van_der_pol_params['y0'] = y0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Van der Pol Oscillator")
    
    return (mu, x0, y0)

def setup_duffing_ui():
    """Setup UI for Duffing oscillator."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        alpha = st.slider("Linear Stiffness (α)", -50.0, 50.0, st.session_state.duffing_params['alpha'], 0.1)
        st.session_state.duffing_params['alpha'] = alpha
        
    with col2:
        alpha = st.number_input("α value", -50.0, 50.0, st.session_state.duffing_params['alpha'], 0.1)
        st.session_state.duffing_params['alpha'] = alpha
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Nonlinear Stiffness (β)", -50.0, 50.0, st.session_state.duffing_params['beta'], 0.1)
        st.session_state.duffing_params['beta'] = beta
        
    with col2:
        beta = st.number_input("β value", -50.0, 50.0, st.session_state.duffing_params['beta'], 0.1)
        st.session_state.duffing_params['beta'] = beta
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        delta = st.slider("Damping (δ)", 0.0, 50.0, st.session_state.duffing_params['delta'], 0.01)
        st.session_state.duffing_params['delta'] = delta
        
    with col2:
        delta = st.number_input("δ value", 0.0, 50.0, st.session_state.duffing_params['delta'], 0.01)
        st.session_state.duffing_params['delta'] = delta
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        gamma = st.slider("Forcing Amplitude (γ)", 0.0, 80.0, st.session_state.duffing_params['gamma'], 0.1)
        st.session_state.duffing_params['gamma'] = gamma
        
    with col2:
        gamma = st.number_input("γ value", 0.0, 80.0, st.session_state.duffing_params['gamma'], 0.1)
        st.session_state.duffing_params['gamma'] = gamma
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        omega = st.slider("Forcing Frequency (ω)", 0.01, 50.0, st.session_state.duffing_params['omega'], 0.01)
        st.session_state.duffing_params['omega'] = omega
        
    with col2:
        omega = st.number_input("ω value", 0.01, 50.0, st.session_state.duffing_params['omega'], 0.01)
        st.session_state.duffing_params['omega'] = omega
    
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.slider("Initial Position (x₀)", -50.0, 50.0, st.session_state.duffing_params['x0'], 0.1)
        st.session_state.duffing_params['x0'] = x0
        
    with col2:
        x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.duffing_params['x0'], 0.1)
        st.session_state.duffing_params['x0'] = x0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        y0 = st.slider("Initial Velocity (y₀)", -50.0, 50.0, st.session_state.duffing_params['y0'], 0.1)
        st.session_state.duffing_params['y0'] = y0
        
    with col2:
        y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.duffing_params['y0'], 0.1)
        st.session_state.duffing_params['y0'] = y0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Duffing Oscillator")
    
    return (alpha, beta, delta, gamma, omega, x0, y0)

def setup_double_pendulum_ui():
    """Setup UI for double pendulum."""
    st.sidebar.subheader("Physical Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        g = st.slider("Gravity (g)", 1.0, 100.0, st.session_state.double_pendulum_params['g'], 0.1)
        st.session_state.double_pendulum_params['g'] = g
        
    with col2:
        g = st.number_input("g value", 1.0, 100.0, st.session_state.double_pendulum_params['g'], 0.1)
        st.session_state.double_pendulum_params['g'] = g
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        m1 = st.slider("Mass 1 (m₁)", 0.1, 10.0, st.session_state.double_pendulum_params['m1'], 0.1)
        st.session_state.double_pendulum_params['m1'] = m1
        
    with col2:
        m1 = st.number_input("m₁ value", 0.1, 10.0, st.session_state.double_pendulum_params['m1'], 0.1)
        st.session_state.double_pendulum_params['m1'] = m1
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        m2 = st.slider("Mass 2 (m₂)", 0.1, 10.0, st.session_state.double_pendulum_params['m2'], 0.1)
        st.session_state.double_pendulum_params['m2'] = m2
        
    with col2:
        m2 = st.number_input("m₂ value", 0.1, 10.0, st.session_state.double_pendulum_params['m2'], 0.1)
        st.session_state.double_pendulum_params['m2'] = m2
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        l1 = st.slider("Length 1 (l₁)", 0.1, 10.0, st.session_state.double_pendulum_params['l1'], 0.1)
        st.session_state.double_pendulum_params['l1'] = l1
        
    with col2:
        l1 = st.number_input("l₁ value", 0.1, 10.0, st.session_state.double_pendulum_params['l1'], 0.1)
        st.session_state.double_pendulum_params['l1'] = l1
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        l2 = st.slider("Length 2 (l₂)", 0.1, 10.0, st.session_state.double_pendulum_params['l2'], 0.1)
        st.session_state.double_pendulum_params['l2'] = l2
        
    with col2:
        l2 = st.number_input("l₂ value", 0.1, 10.0, st.session_state.double_pendulum_params['l2'], 0.1)
        st.session_state.double_pendulum_params['l2'] = l2
        
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theta1 = st.slider("Initial Angle 1 (θ₁, rad)", -np.pi, np.pi, st.session_state.double_pendulum_params['theta1'], 0.1)
        st.session_state.double_pendulum_params['theta1'] = theta1
        
    with col2:
        theta1 = st.number_input("θ₁ value", -np.pi, np.pi, st.session_state.double_pendulum_params['theta1'], 0.1)
        st.session_state.double_pendulum_params['theta1'] = theta1
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        omega1 = st.slider("Initial Angular Velocity 1 (ω₁, rad/s)", -10.0, 10.0, st.session_state.double_pendulum_params['omega1'], 0.1)
        st.session_state.double_pendulum_params['omega1'] = omega1
        
    with col2:
        omega1 = st.number_input("ω₁ value", -10.0, 10.0, st.session_state.double_pendulum_params['omega1'], 0.1)
        st.session_state.double_pendulum_params['omega1'] = omega1
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theta2 = st.slider("Initial Angle 2 (θ₂, rad)", -np.pi, np.pi, st.session_state.double_pendulum_params['theta2'], 0.1)
        st.session_state.double_pendulum_params['theta2'] = theta2
        
    with col2:
        theta2 = st.number_input("θ₂ value", -np.pi, np.pi, st.session_state.double_pendulum_params['theta2'], 0.1)
        st.session_state.double_pendulum_params['theta2'] = theta2
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        omega2 = st.slider("Initial Angular Velocity 2 (ω₂, rad/s)", -10.0, 10.0, st.session_state.double_pendulum_params['omega2'], 0.1)
        st.session_state.double_pendulum_params['omega2'] = omega2
        
    with col2:
        omega2 = st.number_input("ω₂ value", -10.0, 10.0, st.session_state.double_pendulum_params['omega2'], 0.1)
        st.session_state.double_pendulum_params['omega2'] = omega2
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Double Pendulum")
    
    return (g, m1, m2, l1, l2, theta1, omega1, theta2, omega2)

def setup_lorenz_ui():
    """Setup UI for Lorenz system."""
    st.sidebar.subheader("Equation Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sigma = st.slider("Sigma (σ)", 0.1, 100.0, st.session_state.lorenz_params['sigma'], 0.1)
        st.session_state.lorenz_params['sigma'] = sigma
        
    with col2:
        sigma = st.number_input("σ value", 0.1, 100.0, st.session_state.lorenz_params['sigma'], 0.1)
        st.session_state.lorenz_params['sigma'] = sigma
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        rho = st.slider("Rho (ρ)", 0.1, 100.0, st.session_state.lorenz_params['rho'], 0.1)
        st.session_state.lorenz_params['rho'] = rho
        
    with col2:
        rho = st.number_input("ρ value", 0.1, 100.0, st.session_state.lorenz_params['rho'], 0.1)
        st.session_state.lorenz_params['rho'] = rho
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Beta (β)", 0.1, 100.0, st.session_state.lorenz_params['beta'], 0.1)
        st.session_state.lorenz_params['beta'] = beta
        
    with col2:
        beta = st.number_input("β value", 0.1, 100.0, st.session_state.lorenz_params['beta'], 0.1)
        st.session_state.lorenz_params['beta'] = beta
        
    st.sidebar.subheader("Initial Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.slider("Initial x₀", -50.0, 50.0, st.session_state.lorenz_params['x0'], 0.1)
        st.session_state.lorenz_params['x0'] = x0
        
    with col2:
        x0 = st.number_input("x₀ value", -50.0, 50.0, st.session_state.lorenz_params['x0'], 0.1)
        st.session_state.lorenz_params['x0'] = x0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        y0 = st.slider("Initial y₀", -50.0, 50.0, st.session_state.lorenz_params['y0'], 0.1)
        st.session_state.lorenz_params['y0'] = y0
        
    with col2:
        y0 = st.number_input("y₀ value", -50.0, 50.0, st.session_state.lorenz_params['y0'], 0.1)
        st.session_state.lorenz_params['y0'] = y0
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        z0 = st.slider("Initial z₀", -50.0, 50.0, st.session_state.lorenz_params['z0'], 0.1)
        st.session_state.lorenz_params['z0'] = z0
        
    with col2:
        z0 = st.number_input("z₀ value", -50.0, 50.0, st.session_state.lorenz_params['z0'], 0.1)
        st.session_state.lorenz_params['z0'] = z0
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Lorenz System")
    
    return (sigma, rho, beta, x0, y0, z0)

def display_time_point_controls(time_point, max_time_point):
    """Display time point navigation controls."""
    st.subheader("Time Point Analysis")
    
    # Time point selection interface
    time_point = st.slider("Select Time Point", 0, max_time_point, time_point)
    
    # Use horizontal layout for navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("◀ Previous", key="dec_time"):
            time_point = max(0, time_point - 1)
    with col2:
        if st.button("Next ▶", key="inc_time"):
            time_point = min(max_time_point, time_point + 1)
    with col3:
        time_input = st.number_input("Direct input:", min_value=0, 
                                max_value=max_time_point, 
                                value=time_point,
                                step=1)
        if time_input != time_point:
            time_point = time_input
    
    return time_point

def display_feature_controls(feature_idx, max_feature_idx):
    """Display feature selection controls."""
    st.subheader("Feature Exploration")
    
    # Feature selection interface
    feature_idx = st.slider("Select Feature Index", 0, max_feature_idx, feature_idx)
    
    # Use horizontal layout for navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("◀ Previous", key="dec_feature"):
            feature_idx = max(0, feature_idx - 1)
    with col2:
        if st.button("Next ▶", key="inc_feature"):
            feature_idx = min(max_feature_idx, feature_idx + 1)
    with col3:
        feature_input = st.number_input("Direct input:", min_value=0, 
                                  max_value=max_feature_idx, 
                                  value=feature_idx,
                                  step=1)
        if feature_input != feature_idx:
            feature_idx = feature_input
    
    return feature_idx

def display_state_metrics(solution, time_point, system_type, feature_values=None, feature_idx=None):
    """Display metrics about the current system state at the selected time point."""
    st.markdown("**System State:**")
    m_cols = st.columns(4)
    
    # Extract variables based on system type
    if system_type == "Harmonic Oscillator":
        position = solution['solution'][time_point, 0]
        velocity = solution['solution'][time_point, 1]
        params = solution['params']
        omega = params[0]
        gamma = params[1]
        acceleration = -omega**2 * position - gamma * velocity
        
        with m_cols[0]:
            st.metric("Position", f"{position:.4f}")
        with m_cols[1]:
            st.metric("Velocity", f"{velocity:.4f}")
        with m_cols[2]:
            st.metric("Acceleration", f"{acceleration:.4f}")
            
    elif system_type == "Sinusoidal Function":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        params = solution['params']
        amplitude = params[0]
        frequency = params[1]
        phase = params[2]
        use_cos = params[3] if len(params) > 3 else False
        t = solution['time_points'][time_point]
        second_deriv = -amplitude * frequency**2 * (np.sin(frequency * t + phase) if not use_cos else np.cos(frequency * t + phase))
        
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", f"{second_deriv:.4f}")
            
    elif system_type == "Linear Function":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", "0.0000")
            
    elif system_type == "Lotka-Volterra":
        prey = solution['solution'][time_point, 0]
        predator = solution['solution'][time_point, 1]
        params = solution['params']
        alpha = params[0]
        beta = params[1]
        
        with m_cols[0]:
            st.metric("Prey Population", f"{prey:.4f}")
        with m_cols[1]:
            st.metric("Predator Population", f"{predator:.4f}")
        with m_cols[2]:
            # Calculate rate of change
            prey_change = alpha * prey - beta * prey * predator
            st.metric("Prey Rate of Change", f"{prey_change:.4f}")
        
    elif system_type == "FitzHugh-Nagumo":
        v = solution['solution'][time_point, 0]
        w = solution['solution'][time_point, 1]
        params = solution['params']
        I = params[3]
        
        with m_cols[0]:
            st.metric("Membrane Potential (v)", f"{v:.4f}")
        with m_cols[1]:
            st.metric("Recovery Variable (w)", f"{w:.4f}")
        with m_cols[2]:
            # Calculate rate of change
            v_change = v - v**3/3 - w + I
            st.metric("dv/dt", f"{v_change:.4f}")
            
    elif system_type == "Coupled Linear System":
        x = solution['solution'][time_point, 0]
        y = solution['solution'][time_point, 1]
        params = solution['params']
        alpha = params[0]
        beta = params[1]
        
        with m_cols[0]:
            st.metric("x value", f"{x:.4f}")
        with m_cols[1]:
            st.metric("y value", f"{y:.4f}")
        with m_cols[2]:
            # Calculate rates of change
            x_change = alpha * y
            st.metric("dx/dt", f"{x_change:.4f}")
            
    elif system_type == "Van der Pol Oscillator":
        x = solution['solution'][time_point, 0]
        y = solution['solution'][time_point, 1]
        params = solution['params']
        mu = params[0]
        
        with m_cols[0]:
            st.metric("Position (x)", f"{x:.4f}")
        with m_cols[1]:
            st.metric("Velocity (y)", f"{y:.4f}")
        with m_cols[2]:
            # Calculate rate of change
            y_change = mu * (1 - x**2) * y - x
            st.metric("dy/dt", f"{y_change:.4f}")
            
    elif system_type == "Duffing Oscillator":
        x = solution['solution'][time_point, 0]
        y = solution['solution'][time_point, 1]
        t_val = solution['time_points'][time_point]
        params = solution['params']
        alpha = params[0]
        beta = params[1]
        delta = params[2]
        gamma = params[3]
        omega = params[4]
        
        with m_cols[0]:
            st.metric("Position (x)", f"{x:.4f}")
        with m_cols[1]:
            st.metric("Velocity (y)", f"{y:.4f}")
        with m_cols[2]:
            # Calculate rate of change
            y_change = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t_val)
            st.metric("dy/dt", f"{y_change:.4f}")
            
    elif system_type == "Double Pendulum":
        theta1 = solution['solution'][time_point, 0]
        omega1 = solution['solution'][time_point, 1]
        theta2 = solution['solution'][time_point, 2]
        omega2 = solution['solution'][time_point, 3]
        
        with m_cols[0]:
            st.metric("θ₁", f"{theta1:.4f}")
        with m_cols[1]:
            st.metric("ω₁", f"{omega1:.4f}")
        with m_cols[2]:
            st.metric("θ₂", f"{theta2:.4f}")
        with m_cols[3]:
            st.metric("ω₂", f"{omega2:.4f}")
            
    elif system_type == "Lorenz System":
        x = solution['solution'][time_point, 0]
        y = solution['solution'][time_point, 1]
        z = solution['solution'][time_point, 2]
        params = solution['params']
        sigma = params[0]
        
        with m_cols[0]:
            st.metric("x", f"{x:.4f}")
        with m_cols[1]:
            st.metric("y", f"{y:.4f}")
        with m_cols[2]:
            st.metric("z", f"{z:.4f}")
        with m_cols[3]:
            # Calculate one of the rates of change
            x_change = sigma * (y - x)
            st.metric("dx/dt", f"{x_change:.4f}")
    elif system_type == "Simple Exponential":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        params = solution['params']
        a, b, c = params
        t = solution['time_points'][time_point]
        second_deriv = a * b * b * np.exp(b * t)
        
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", f"{second_deriv:.4f}")
            
    elif system_type == "Simple Polynomial":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        params = solution['params']
        a, b, c = params
        t = solution['time_points'][time_point]
        
        # Calculate second derivative: a * b * (b-1) * t^(b-2)
        if b >= 2 or b <= 0:  # Avoid division by zero or negative powers
            second_deriv = a * b * (b-1) * np.power(t, b-2) if t > 0 else np.nan
        else:
            second_deriv = 0
            
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", f"{second_deriv:.4f}" if not np.isnan(second_deriv) else "undefined")
            
    elif system_type == "Sigmoid Function":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        params = solution['params']
        a, b, c = params
        t = solution['time_points'][time_point]
        
        # Calculate second derivative: b^2 * value * (1 - value/a) * (1 - 2*value/a)
        second_deriv = b * b * value * (1 - value/a) * (1 - 2*value/a)
        
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", f"{second_deriv:.4f}")
            
    elif system_type == "Tanh Function":
        value = solution['solution'][time_point, 0]
        derivative = solution['solution'][time_point, 1]
        params = solution['params']
        a, b, c = params
        t = solution['time_points'][time_point]
        
        # Calculate second derivative: -2 * a * b^2 * tanh(b*(t-c)) * (1 - tanh^2(b*(t-c)))
        tanh_val = np.tanh(b * (t - c))
        second_deriv = -2 * a * b * b * tanh_val * (1 - tanh_val**2)
        
        with m_cols[0]:
            st.metric("Value", f"{value:.4f}")
        with m_cols[1]:
            st.metric("Derivative", f"{derivative:.4f}")
        with m_cols[2]:
            st.metric("Second Derivative", f"{second_deriv:.4f}")
    
    # Show selected feature value if available
    if feature_values is not None and feature_idx is not None:
        st.metric(f"Feature {feature_idx} value", f"{feature_values[time_point]:.4f}")


def setup_exponential_ui():
    """Setup UI for exponential function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Coefficient (a)", -50.0, 50.0, st.session_state.get('exp_a', 1.0), 0.1)
        st.session_state.exp_a = a
        
    with col2:
        a = st.number_input("a value", -50.0, 50.0, st.session_state.exp_a, 0.1, key="exp_a_input")
        st.session_state.exp_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Exponent Coefficient (b)", -5.0, 5.0, st.session_state.get('exp_b', 0.5), 0.1)
        st.session_state.exp_b = b
        
    with col2:
        b = st.number_input("b value", -5.0, 5.0, st.session_state.exp_b, 0.1, key="exp_b_input")
        st.session_state.exp_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Offset (c)", -50.0, 50.0, st.session_state.get('exp_c', 0.0), 0.1)
        st.session_state.exp_c = c
        
    with col2:
        c = st.number_input("c value", -50.0, 50.0, st.session_state.exp_c, 0.1, key="exp_c_input")
        st.session_state.exp_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Simple Exponential")
    
    return (a, b, c)

def setup_polynomial_ui():
    """Setup UI for polynomial function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Coefficient (a)", -50.0, 50.0, st.session_state.get('poly_a', 1.0), 0.1)
        st.session_state.poly_a = a
        
    with col2:
        a = st.number_input("a value", -50.0, 50.0, st.session_state.poly_a, 0.1, key="poly_a_input")
        st.session_state.poly_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Power (b)", -5.0, 5.0, st.session_state.get('poly_b', 2.0), 0.1)
        st.session_state.poly_b = b
        
    with col2:
        b = st.number_input("b value", -5.0, 5.0, st.session_state.poly_b, 0.1, key="poly_b_input")
        st.session_state.poly_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Offset (c)", -50.0, 50.0, st.session_state.get('poly_c', 0.0), 0.1)
        st.session_state.poly_c = c
        
    with col2:
        c = st.number_input("c value", -50.0, 50.0, st.session_state.poly_c, 0.1, key="poly_c_input")
        st.session_state.poly_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Simple Polynomial")
    
    return (a, b, c)

def setup_sigmoid_ui():
    """Setup UI for sigmoid function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Amplitude (a)", 0.1, 50.0, st.session_state.get('sigmoid_a', 1.0), 0.1)
        st.session_state.sigmoid_a = a
        
    with col2:
        a = st.number_input("a value", 0.1, 50.0, st.session_state.sigmoid_a, 0.1, key="sigmoid_a_input")
        st.session_state.sigmoid_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Steepness (b)", 0.1, 10.0, st.session_state.get('sigmoid_b', 1.0), 0.1)
        st.session_state.sigmoid_b = b
        
    with col2:
        b = st.number_input("b value", 0.1, 10.0, st.session_state.sigmoid_b, 0.1, key="sigmoid_b_input")
        st.session_state.sigmoid_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Center (c)", -10.0, 10.0, st.session_state.get('sigmoid_c', 0.0), 0.1)
        st.session_state.sigmoid_c = c
        
    with col2:
        c = st.number_input("c value", -10.0, 10.0, st.session_state.sigmoid_c, 0.1, key="sigmoid_c_input")
        st.session_state.sigmoid_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Sigmoid Function")
    
    return (a, b, c)

def setup_tanh_ui():
    """Setup UI for tanh function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Amplitude (a)", 0.1, 50.0, st.session_state.get('tanh_a', 1.0), 0.1)
        st.session_state.tanh_a = a
        
    with col2:
        a = st.number_input("a value", 0.1, 50.0, st.session_state.tanh_a, 0.1, key="tanh_a_input")
        st.session_state.tanh_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Steepness (b)", 0.1, 10.0, st.session_state.get('tanh_b', 1.0), 0.1)
        st.session_state.tanh_b = b
        
    with col2:
        b = st.number_input("b value", 0.1, 10.0, st.session_state.tanh_b, 0.1, key="tanh_b_input")
        st.session_state.tanh_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Center (c)", -10.0, 10.0, st.session_state.get('tanh_c', 0.0), 0.1)
        st.session_state.tanh_c = c
        
    with col2:
        c = st.number_input("c value", -10.0, 10.0, st.session_state.tanh_c, 0.1, key="tanh_c_input")
        st.session_state.tanh_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Tanh Function")
    
    return (a, b, c)

def setup_polynomial_ui():
    """Setup UI for polynomial function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Coefficient (a)", -50.0, 50.0, st.session_state.get('poly_a', 1.0), 0.1)
        st.session_state.poly_a = a
        
    with col2:
        a = st.number_input("a value", -50.0, 50.0, st.session_state.poly_a, 0.1, key="poly_a_input")
        st.session_state.poly_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Power (b)", -5.0, 5.0, st.session_state.get('poly_b', 2.0), 0.1)
        st.session_state.poly_b = b
        
    with col2:
        b = st.number_input("b value", -5.0, 5.0, st.session_state.poly_b, 0.1, key="poly_b_input")
        st.session_state.poly_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Offset (c)", -50.0, 50.0, st.session_state.get('poly_c', 0.0), 0.1)
        st.session_state.poly_c = c
        
    with col2:
        c = st.number_input("c value", -50.0, 50.0, st.session_state.poly_c, 0.1, key="poly_c_input")
        st.session_state.poly_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Simple Polynomial")
    
    return (a, b, c)

def setup_sigmoid_ui():
    """Setup UI for sigmoid function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Amplitude (a)", 0.1, 50.0, st.session_state.get('sigmoid_a', 1.0), 0.1)
        st.session_state.sigmoid_a = a
        
    with col2:
        a = st.number_input("a value", 0.1, 50.0, st.session_state.sigmoid_a, 0.1, key="sigmoid_a_input")
        st.session_state.sigmoid_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Steepness (b)", 0.1, 10.0, st.session_state.get('sigmoid_b', 1.0), 0.1)
        st.session_state.sigmoid_b = b
        
    with col2:
        b = st.number_input("b value", 0.1, 10.0, st.session_state.sigmoid_b, 0.1, key="sigmoid_b_input")
        st.session_state.sigmoid_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Center (c)", -10.0, 10.0, st.session_state.get('sigmoid_c', 0.0), 0.1)
        st.session_state.sigmoid_c = c
        
    with col2:
        c = st.number_input("c value", -10.0, 10.0, st.session_state.sigmoid_c, 0.1, key="sigmoid_c_input")
        st.session_state.sigmoid_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Sigmoid Function")
    
    return (a, b, c)

def setup_tanh_ui():
    """Setup UI for tanh function."""
    st.sidebar.subheader("Function Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.slider("Amplitude (a)", 0.1, 50.0, st.session_state.get('tanh_a', 1.0), 0.1)
        st.session_state.tanh_a = a
        
    with col2:
        a = st.number_input("a value", 0.1, 50.0, st.session_state.tanh_a, 0.1, key="tanh_a_input")
        st.session_state.tanh_a = a
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        b = st.slider("Steepness (b)", 0.1, 10.0, st.session_state.get('tanh_b', 1.0), 0.1)
        st.session_state.tanh_b = b
        
    with col2:
        b = st.number_input("b value", 0.1, 10.0, st.session_state.tanh_b, 0.1, key="tanh_b_input")
        st.session_state.tanh_b = b
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        c = st.slider("Center (c)", -10.0, 10.0, st.session_state.get('tanh_c', 0.0), 0.1)
        st.session_state.tanh_c = c
        
    with col2:
        c = st.number_input("c value", -10.0, 10.0, st.session_state.tanh_c, 0.1, key="tanh_c_input")
        st.session_state.tanh_c = c
    
    # Add Poincaré section UI
    setup_poincare_section_ui("Tanh Function")
    
    return (a, b, c)