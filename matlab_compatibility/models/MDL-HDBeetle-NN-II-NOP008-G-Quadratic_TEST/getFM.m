function [F, M] = getFM(state, omega, modelParams, droneParams)
	% Model ID: MDL-HDBeetle-NN-II-NOP008-G-Quadratic_TEST
	% Model specific parameters should be defined in the workspace
	% -> Make sure <modelParams> and <droneParams> are loaded for MDL-HDBeetle-NN-II-NOP008-G-Quadratic_TEST
	% Otherwise, run model_init(MDL-HDBeetle-NN-II-NOP008-G-Quadratic_TEST)


	% Get input vector for quadrotor models
	droneInputs = getDroneInputs(state, omega, modelParams, droneParams);

	% Compute forces and moments
	Fx = get_Fx(droneInputs.u, droneInputs.w_tot, droneInputs.pitch, droneInputs.w, droneInputs.mu_z);
	Fy = get_Fy(droneInputs.v, droneInputs.w_tot, droneInputs.p, droneInputs.w, droneInputs.roll);
	Fz = get_Fz(droneInputs.w, droneInputs.w2_1, droneInputs.w2_2, droneInputs.w2_3, droneInputs.w2_4, droneInputs.u, droneInputs.U_p, droneInputs.v, droneInputs.pitch);
	Mx = get_Mx(droneInputs.p, droneInputs.U_p, droneInputs.r, droneInputs.q);
	My = get_My(droneInputs.q, droneInputs.U_q);
	Mz = get_Mz(droneInputs.r, droneInputs.U_r, droneInputs.p, droneInputs.q);

	% Collect forces, F, and moments, M
	F = [Fx,Fy,Fz] ./ droneInputs.Fden;
	M = [Mx,My,Mz] ./ droneInputs.Mden;
