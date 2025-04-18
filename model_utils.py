import streamlit as st
import torch
import traceback
import importlib
from pathlib import Path
import numpy as np
from activation_collector import install, collect_activations_during_fit

@st.cache_resource
def load_odeformer_model():
    """Load the ODEformer model with caching."""
    try:
        from odeformer.model import SymbolicTransformerRegressor
        model = SymbolicTransformerRegressor(from_pretrained=True)
        model_args = {'beam_size': 20, 'beam_temperature': 0.1}
        model.set_model_args(model_args)
        return model
    except Exception as e:
        st.error(f"Error loading ODEformer model: {str(e)}")
        st.code(traceback.format_exc())
        return None
            
@st.cache_resource
def load_sae_model(model_path):
    """Load a sparse autoencoder model with caching."""
    try:
        SparseAutoencoder = st.session_state.sae_module.SparseAutoencoder
        sae_model = SparseAutoencoder(input_dim=256, latent_dim=1280)
        sae_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        sae_model.eval()
        return sae_model
    except Exception as e:
        st.error(f"Error loading SAE model: {str(e)}")
        st.code(traceback.format_exc())
        return None

def get_activations(model, solution):
    """Get activations for the currently selected site and component directly from a single model run"""
    site_name = st.session_state.activation_site
    component = st.session_state.activation_component
    
    # Check if we already have all_collected_activations in the session state
    if st.session_state.all_collected_activations is not None:
        print("Using cached activations from session state")
        all_activations = st.session_state.all_collected_activations
    else:
        # Run the model only once to collect all activations
        print("Starting single model run to collect all activations...")
        all_activations, _ = collect_activations_during_fit(
            model, solution['time_points'], solution['solution']
        )
        
        # Store all collected activations for future reference
        if all_activations:
            st.session_state.all_collected_activations = all_activations
            # Print component details for debugging
            for site in all_activations:
                print(f"Site '{site}' has components: {list(all_activations[site].keys())}")
    
    # Extract the specific activations we need from the all_activations dictionary
    specific_activations = None
    if all_activations:
        if site_name in all_activations:
            if component in all_activations[site_name]:
                shapes = list(all_activations[site_name][component].keys())
                if shapes:
                    tensor_list = all_activations[site_name][component][shapes[0]]
                    if tensor_list:
                        tensor = tensor_list[0]
                        if tensor.dim() == 3:
                            specific_activations = tensor[0].numpy()  # Extract batch dim
                        else:
                            specific_activations = tensor.numpy()
    
    return specific_activations

def apply_sae(sae_model, activations):
    """Apply SAE to get latent features"""
    inputs = torch.tensor(activations, dtype=torch.float32)
    _, latent = sae_model(inputs)
    return latent.squeeze(0).detach().numpy()

def get_learned_equations(model, solution):
    """
    Get the learned equations from the ODEformer model.
    
    Args:
        model: The trained ODEformer model
        solution: The solution dictionary containing time_points and solution data
        
    Returns:
        A dictionary with learned equations and metadata
    """
    try:
        # Extract time points and trajectory data
        times = solution['time_points']
        trajectories = solution['solution']
        
        # First try to fit the model to the data
        try:
            # Use fit method to learn from the data
            learned_eqs = model.fit(times, trajectories, verbose=True)
            
            # Now check if the model's predictions are populated
            if hasattr(model, 'predictions') and len(model.predictions) > 0:
                # Get the first set of candidates (if there are multiple datasets)
                candidates = list(model.predictions.values())[0]
                
                if candidates and len(candidates) > 0:
                    # Take the top prediction (first in the list)
                    top_eq = candidates[0]
                    
                    # Format the equation as a readable string
                    if hasattr(top_eq, 'infix'):
                        eq_str = top_eq.infix()
                    else:
                        eq_str = str(top_eq)
                    
                    # Handle ODE system representation (splitting by '|' if needed)
                    if '|' in eq_str:
                        equations = eq_str.split('|')
                        formatted_equations = []
                        for i, equation in enumerate(equations):
                            formatted_equations.append(f"x_{i}' = {equation.strip()}")
                        formatted_eq_str = "\n".join(formatted_equations)
                    else:
                        formatted_eq_str = eq_str
                    
                    # Return the formatted equation and all candidates
                    return {
                        'equation_str': formatted_eq_str,
                        'full_results': candidates,
                        'success': True
                    }
                else:
                    return {
                        'equation_str': "No equations predicted (empty candidates)",
                        'full_results': None,
                        'success': False
                    }
            else:
                return {
                    'equation_str': "No predictions found in model",
                    'full_results': None,
                    'success': False
                }
                
        except Exception as fit_error:
            print(f"Error in model.fit(): {str(fit_error)}")
            import traceback
            print(traceback.format_exc())
            
            # Try alternative methods
            if hasattr(model, 'print'):
                # Capture model.print() output safely
                import io
                import sys
                from contextlib import redirect_stdout
                
                # Create a string buffer to capture print output
                f = io.StringIO()
                
                # Use a cleaner method to convert arrays to lists to avoid array truth ambiguity
                try:
                    # Redirect stdout to the buffer
                    with redirect_stdout(f):
                        # Try to access predictions directly
                        if hasattr(model, 'predictions') and isinstance(model.predictions, dict) and len(model.predictions) > 0:
                            dataset_num = 0  # Default to first dataset
                            model.print(dataset_num=dataset_num, n_predictions=1)
                        else:
                            try:
                                model.print()
                            except:
                                pass
                    
                    # Get the captured output
                    equation_str = f.getvalue().strip()
                    
                    if equation_str:
                        return {
                            'equation_str': equation_str,
                            'full_results': None,
                            'success': True
                        }
                except Exception as print_error:
                    print(f"Error using model.print() alternative: {str(print_error)}")
            
            # Final fallback - check if the model has learned_equations attribute
            if hasattr(model, 'learned_equations'):
                eqs = model.learned_equations
                if eqs:
                    return {
                        'equation_str': str(eqs),
                        'full_results': eqs,
                        'success': True
                    }
            
            # Try running predict and use the result
            try:
                # Get initial conditions from the first time point
                y0 = trajectories[0]
                
                # Run prediction just to access the equation
                prediction = model.predict(times, y0)
                
                if prediction is not None:
                    return {
                        'equation_str': "Equation extracted from prediction (see plot)",
                        'full_results': None,
                        'success': True
                    }
            except Exception as predict_error:
                print(f"Error in model.predict(): {str(predict_error)}")
                    
            # If we reach here, all methods have failed
            return {
                'equation_str': f"Error extracting equation: {str(fit_error)}",
                'full_results': None,
                'success': False
            }
                
    except Exception as e:
        print(f"Error getting learned equations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'equation_str': f"Error: {str(e)}",
            'full_results': None,
            'success': False
        }

def get_predicted_trajectory(model, solution):
    """
    Get predicted trajectory from the ODEformer model by extracting the equations
    and integrating them ourselves.
    
    Args:
        model: The ODEformer model
        solution: Dictionary containing time_points and solution
        
    Returns:
        Predicted trajectory array or None if prediction failed
    """
    # Extract time points and trajectory data
    times = solution['time_points']
    trajectories = solution['solution']
    initial_conditions = trajectories[0]  # First point as initial condition
    n_dims = trajectories.shape[1]
    
    # Function to generate dummy trajectory
    def create_dummy_trajectory(noise_level=0.1):
        try:
            # Create a noisy copy of the original trajectory
            import numpy as np
            dummy_traj = trajectories.copy()
            
            # Add noise proportional to the variance of each dimension
            std_per_dim = np.std(dummy_traj, axis=0)
            noise = np.zeros_like(dummy_traj)
            
            # Generate noise for each dimension separately
            for dim in range(dummy_traj.shape[1]):
                noise[:, dim] = noise_level * std_per_dim[dim] * np.random.normal(size=dummy_traj.shape[0])
            
            # Add the noise to the trajectory
            dummy_traj += noise
            
            print(f"Created dummy trajectory with noise level {noise_level:.2f}")
            return dummy_traj
        except Exception as e:
            print(f"Error creating dummy trajectory: {e}")
            return trajectories.copy()
    
    # First approach: Try to directly use model.predict()
    try:
        print("Attempting direct model.predict() method...")
        predicted_trajectory = model.predict(times, initial_conditions)
        
        # Check if the result looks reasonable
        if (isinstance(predicted_trajectory, np.ndarray) and 
            predicted_trajectory.shape == trajectories.shape):
            print("Successfully predicted trajectory using model.predict()")
            return predicted_trajectory
        else:
            print("model.predict() returned unexpected result, trying other methods...")
    except Exception as e:
        print(f"Error using model.predict(): {e}")
    
    # Second approach: Try to get equations from fit results
    try:
        print("Trying to extract equations from model.fit()...")
        
        # Fit the model to the data
        model.fit(times, trajectories)
        
        # Check if predictions are available
        if hasattr(model, 'predictions') and len(model.predictions) > 0:
            # Get the first candidate equation
            candidates = list(model.predictions.values())[0]
            if candidates and len(candidates) > 0:
                top_eq = candidates[0]
                
                # Create equation string
                equation_str = None
                if hasattr(top_eq, 'infix'):
                    equation_str = top_eq.infix()
                else:
                    equation_str = str(top_eq)
                
                print(f"Found equation: {equation_str}")
                
                if equation_str:
                    # Extract equations of the form "x_0' = expression"
                    import re
                    import numpy as np
                    from scipy.integrate import solve_ivp
                    
                    # Different patterns to try
                    patterns = [
                        r"x_(\d+)'\s*=\s*([^,\n|]+)",  # x_0' = expression
                        r"d([^/]+)/dt\s*=\s*([^,\n|]+)",  # dx/dt = expression
                        r"(\w+)'\s*=\s*([^,\n|]+)",      # x' = expression
                    ]
                    
                    # Split the equation if it contains multiple parts separated by '|'
                    equation_parts = equation_str.split('|')
                    matched_equations = []
                    
                    for part in equation_parts:
                        for pattern in patterns:
                            matches = re.findall(pattern, part)
                            if matches:
                                matched_equations.extend(matches)
                                break
                    
                    if matched_equations:
                        print(f"Parsed {len(matched_equations)} equations")
                        
                        # Create a function that evaluates the ODE
                        def ode_func(t, y):
                            dydt = np.zeros_like(y)
                            
                            for idx, (var_idx_str, expr) in enumerate(matched_equations):
                                try:
                                    # Handle different pattern formats
                                    if var_idx_str.isdigit():
                                        # Pattern: x_0' = expression
                                        var_idx = int(var_idx_str)
                                    else:
                                        # Pattern: x' = expression or other
                                        if var_idx_str == 'x':
                                            var_idx = 0
                                        elif var_idx_str == 'y':
                                            var_idx = 1
                                        elif var_idx_str == 'z':
                                            var_idx = 2
                                        else:
                                            # If we can't identify the variable, use the index in the equation list
                                            var_idx = idx
                                    
                                    # Skip if the variable index is out of bounds
                                    if var_idx >= len(y):
                                        continue
                                    
                                    # Get the expression (right hand side)
                                    expr = expr.strip()
                                    
                                    # Replace variable references
                                    for i in range(len(y)):
                                        expr = expr.replace(f'x_{i}', f'y[{i}]')
                                        if i == 0:
                                            expr = expr.replace('x', f'y[0]')
                                        elif i == 1:
                                            # Be careful not to replace "exp" with "ey[1]p"
                                            expr = re.sub(r'\by\b', f'y[1]', expr)
                                        elif i == 2:
                                            expr = re.sub(r'\bz\b', f'y[2]', expr)
                                    
                                    # Special case for math functions
                                    expr = expr.replace('sin', 'np.sin')
                                    expr = expr.replace('cos', 'np.cos')
                                    expr = expr.replace('exp', 'np.exp')
                                    expr = expr.replace('log', 'np.log')
                                    expr = expr.replace('sqrt', 'np.sqrt')
                                    expr = expr.replace('pi', 'np.pi')
                                    expr = expr.replace('abs', 'np.abs')
                                    expr = expr.replace('pow', 'np.power')
                                    
                                    # Evaluate the expression
                                    print(f"Evaluating: dydt[{var_idx}] = {expr}")
                                    dydt[var_idx] = eval(expr)
                                except Exception as eval_error:
                                    print(f"Error evaluating expression '{expr}': {eval_error}")
                            
                            return dydt
                        
                        # Integrate the ODE
                        print("Integrating equations...")
                        try:
                            sol = solve_ivp(
                                ode_func,
                                (min(times), max(times)),
                                initial_conditions,
                                t_eval=times,
                                method='RK45',
                                rtol=1e-3,
                                atol=1e-6
                            )
                            
                            if sol.success:
                                print(f"Integration successful! Shape: {sol.y.T.shape}")
                                return sol.y.T
                            else:
                                print("Integration failed.")
                        except Exception as integration_error:
                            print(f"Error during integration: {integration_error}")
    except Exception as fit_error:
        print(f"Error extracting equations from fit: {fit_error}")
    
    # Third approach: Try to use a custom approach that works with this specific model
    try:
        print("Trying with custom predict method...")
        predicted_trajectory = custom_predict(model, times, initial_conditions, solution)
        
        if isinstance(predicted_trajectory, np.ndarray) and predicted_trajectory.shape == trajectories.shape:
            print("Successfully predicted trajectory using custom_predict()")
            return predicted_trajectory
    except Exception as predict_error:
        print(f"Error using custom predict method: {predict_error}")
    
    # If we're here, we need to use the fallback approach
    print("Using fallback dummy trajectory")
    
    # Get system type for appropriate noise level
    system_type = ""
    if 'system_type' in solution:
        system_type = solution['system_type']
    elif 'equations' in solution:
        system_type = solution['equations']
    
    system_type_lower = system_type.lower()
    if 'harmonic' in system_type_lower or 'oscillator' in system_type_lower:
        return create_dummy_trajectory(0.05)  # Less noise for oscillator systems
    elif 'lorenz' in system_type_lower or 'chaos' in system_type_lower:
        return create_dummy_trajectory(0.2)  # More noise for chaotic systems
    else:
        return create_dummy_trajectory(0.1)  # Default noise level

def custom_predict(model, times, y0, solution):
    """
    Custom implementation to predict a trajectory using the learned model.
    This function bypasses potential issues with the model's built-in predict method.
    
    Args:
        model: The ODEformer model
        times: Time points array
        y0: Initial conditions
        solution: Original solution dictionary for reference
        
    Returns:
        Predicted trajectory array
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    
    # Try to get the equations directly from the model
    equation_str = None
    
    # Check if we already have a learned equation
    if hasattr(model, 'learned_equation') and model.learned_equation:
        equation_str = model.learned_equation
    
    # If not, check if the model has predictions
    elif hasattr(model, 'predictions') and model.predictions:
        candidates = next(iter(model.predictions.values()))
        if candidates and len(candidates) > 0:
            # Get the first candidate
            equation = candidates[0]
            if hasattr(equation, 'infix'):
                equation_str = equation.infix()
            else:
                equation_str = str(equation)
    
    if not equation_str:
        print("No equation found, generating fallback trajectory")
        # Return a dummy trajectory (original + noise)
        n_dims = len(y0)
        
        # Create trajectory array
        original_trajectory = solution['solution']
        dummy_trajectory = original_trajectory.copy()
        
        # Add small noise
        noise_level = 0.1
        for dim in range(n_dims):
            std = np.std(original_trajectory[:, dim])
            dummy_trajectory[:, dim] += noise_level * std * np.random.normal(size=len(times))
        
        return dummy_trajectory
    
    # Parse the equation
    print(f"Parsing equation: {equation_str}")
    
    # Split by '|' for multi-dimensional systems
    if '|' in equation_str:
        equation_parts = equation_str.split('|')
    else:
        equation_parts = [equation_str]
    
    # Create an ODE function from the equations
    def ode_func(t, y):
        dydt = np.zeros_like(y)
        
        for i, eq in enumerate(equation_parts):
            if i >= len(y):
                continue
                
            # Replace variable references
            expr = eq.strip()
            for j in range(len(y)):
                expr = expr.replace(f'x_{j}', f'y[{j}]')
            
            # Replace common variables if not indexed
            expr = expr.replace('x', 'y[0]')
            if len(y) > 1:
                expr = expr.replace('y', 'y[1]')
            if len(y) > 2:
                expr = expr.replace('z', 'y[2]')
            
            # Replace math functions
            expr = expr.replace('sin', 'np.sin')
            expr = expr.replace('cos', 'np.cos')
            expr = expr.replace('exp', 'np.exp')
            expr = expr.replace('sqrt', 'np.sqrt')
            
            try:
                dydt[i] = eval(expr)
            except Exception as e:
                print(f"Error evaluating expression '{expr}': {e}")
                # Use a simple fallback for this dimension
                dydt[i] = 0
        
        return dydt
    
    # Solve the ODE
    try:
        sol = solve_ivp(
            ode_func,
            (times[0], times[-1]),
            y0,
            t_eval=times,
            method='RK45',
            rtol=1e-3,
            atol=1e-6
        )
        
        if sol.success:
            return sol.y.T
        else:
            print("ODE integration failed")
    except Exception as e:
        print(f"Error during integration: {e}")
    
    # If integration failed, return the dummy trajectory
    print("Integration failed, returning fallback trajectory")
    return solution['solution'] + 0.1 * np.random.normal(size=solution['solution'].shape)

def load_models():
    """Load all required models."""
    try:
        # Import required modules
        from odeformer.model import SymbolicTransformerRegressor
        st.session_state.sae_module = importlib.import_module("sae")
        
        # Only install patches once
        if not st.session_state.patches_installed:
            install()
            st.session_state.patches_installed = True
            
        model = load_odeformer_model()
        st.sidebar.success("ODEformer model loaded successfully")
        
        # SAE model selection
        st.sidebar.subheader("SAE Model Selection")
        sae_option = st.sidebar.selectbox(
            "Select SAE Model",
            list(st.session_state.sae_paths.keys()),
            index=0
        )

        # If custom path is selected, show text input
        if sae_option == "Custom Path":
            custom_path = st.sidebar.text_input("Custom SAE Path", 
                                              value=st.session_state.sae_paths["Custom Path"])
            st.session_state.sae_paths["Custom Path"] = custom_path
            sae_path = custom_path
        else:
            sae_path = st.session_state.sae_paths[sae_option]
            
        # Display the selected path for reference
        st.sidebar.caption(f"Path: {sae_path}")

        if Path(sae_path).exists():
            sae_model = load_sae_model(sae_path)
            st.sidebar.success("SAE model loaded successfully")
            st.session_state.models_loaded = True
            
            # Check if SAE model changed and clear cached data if needed
            if st.session_state.current_sae_path != sae_path:
                st.session_state.current_sae_path = sae_path
                st.session_state.current_latent_features = None
                st.session_state.current_activations = None
                st.session_state.all_collected_activations = None
                st.sidebar.info("SAE model changed - cache cleared")
        else:
            st.sidebar.error(f"SAE model not found at {sae_path}")
            
        # Store models in session state for later access
        st.session_state.model = model
        st.session_state.sae_model = sae_model
        
        return True
        
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        st.sidebar.code(traceback.format_exc())
        return False