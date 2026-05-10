import numpy as np
import matplotlib.pyplot as plt

import sim_T as sim


REAL_OVERLAP = np.array([845, 822, 718, 656, 650, 595, 575, 558], dtype=float)
REAL_NON_OVERLAP = np.array([830, 815, 705, 645, 630, 595, 568, 558], dtype=float)


def build_parameter_set(set_id: int) -> sim.parameter_change:
	params = sim.parameter_change(set_id)
	if set_id == 0:
		params.xs_hc0 = 1.0
		params.xs_hc1 = 1.0
		params.view_factor = 1.0
		params.xs_tauf = 1.0
		params.xs_taup = 1.0
		params.xs_dqp = 1.0
		params.xs_dqf = 1.0
	elif set_id == 1:
		params = sim.build_parameter_from_file(set_id)
	else:
		raise ValueError(f"Unsupported parameter set id: {set_id}")
	return params


def reset_simulation_state():
	sm = sim.simulation_model
	n = sm.N

	sm.f_total_0 = np.zeros(n)
	sm.f_total_1 = np.zeros(n)

	sm.suma_f0 = np.zeros(n)
	sm.mark_sf_0 = np.zeros(n, dtype=int)
	sm.mark_ef_0 = np.zeros(n, dtype=int)
	sm.suma_f1 = np.zeros(n)
	sm.mark_sf_1 = np.zeros(n, dtype=int)
	sm.mark_ef_1 = np.zeros(n, dtype=int)

	sm.suma_p0 = np.zeros(n)
	sm.mark_sp_0 = np.zeros(n, dtype=int)
	sm.mark_ep_0 = np.zeros(n, dtype=int)
	sm.suma_p1 = np.zeros(n)
	sm.mark_sp_1 = np.zeros(n, dtype=int)
	sm.mark_ep_1 = np.zeros(n, dtype=int)

	sm.history_time = []
	sm.history_T_0 = [[] for _ in range(n)]
	sm.history_T_1 = [[] for _ in range(n)]
	sm.history_Q_0 = [[] for _ in range(n)]
	sm.history_Q_1 = [[] for _ in range(n)]
	sm.history_h_0 = [[] for _ in range(3)]
	sm.history_h_1 = [[] for _ in range(3)]
	sm.pearlite_0 = [[] for _ in range(n)]
	sm.pearlite_1 = [[] for _ in range(n)]


def run_simulation_with_params(params: sim.parameter_change):
	sm = sim.simulation_model
	reset_simulation_state()
	sm.current_params = params

	rolls, num_rolls = sim.data_loader.load_roll_data()
	temp_trans_1 = np.full(sm.N, 850.0)
	temp_trans_0 = np.full(sm.N, 830.0)

	sim.each_roll_time = 0.0
	sim.roll_start_time = [sim.each_roll_time]

	for i in range(num_rolls):
		current_roll = rolls[i]
		current_roll.pre_temp_0 = temp_trans_0
		current_roll.pre_temp_1 = temp_trans_1
		sm.Cooling_calculation(current_roll)
		temp_trans_0 = current_roll.post_temp_0
		temp_trans_1 = current_roll.post_temp_1
		sim.roll_start_time.append(sim.each_roll_time)

	pred_overlap, pred_non_overlap = sm.get_measure_point_T_results()
	return np.asarray(pred_overlap), np.asarray(pred_non_overlap)


def calc_metrics(pred_overlap, pred_non_overlap):
	y_true = np.concatenate([REAL_OVERLAP, REAL_NON_OVERLAP])
	y_pred = np.concatenate([pred_overlap, pred_non_overlap])
	abs_err = np.abs(y_pred - y_true)

	mse = np.mean((y_pred - y_true) ** 2)
	mae = np.mean(abs_err)
	max_dev = np.max(abs_err)

	true_range = np.max(y_true) - np.min(y_true)
	max_dev_pct = (max_dev / true_range) * 100 if true_range > 0 else np.nan

	return {
		"mse": mse,
		"mae": mae,
		"max_dev": max_dev,
		"max_dev_pct": max_dev_pct,
	}


def plot_parity(ax, pred_overlap, pred_non_overlap, title, metrics):
	y_true = np.concatenate([REAL_OVERLAP, REAL_NON_OVERLAP])
	y_pred = np.concatenate([pred_overlap, pred_non_overlap])

	x_min = min(np.min(y_true), np.min(y_pred))
	x_max = max(np.max(y_true), np.max(y_pred))
	span = x_max - x_min
	padding = span * 0.05 if span > 0 else 1.0
	x0 = x_min - padding
	x1 = x_max + padding
	x_line = np.linspace(x0, x1, 200)

	max_dev = metrics["max_dev"]
	mae = metrics["mae"]

	ax.scatter(REAL_OVERLAP, pred_overlap, color="#1f77b4", s=40, label="Overlap")
	ax.scatter(REAL_NON_OVERLAP, pred_non_overlap, color="#ff7f0e", s=40, label="Non-overlap")

	ax.plot(x_line, x_line, color="black", linewidth=1.5, label="y = x")
	ax.plot(x_line, x_line + max_dev, "--", color="#d62728", linewidth=1.2, label=f"+max dev ({max_dev:.2f})")
	ax.plot(x_line, x_line - max_dev, "--", color="#d62728", linewidth=1.2, label=f"-max dev ({max_dev:.2f})")
	ax.plot(x_line, x_line + mae, "--", color="#2ca02c", linewidth=1.2, label=f"+MAE ({mae:.2f})")
	ax.plot(x_line, x_line - mae, "--", color="#2ca02c", linewidth=1.2, label=f"-MAE ({mae:.2f})")

	ax.set_xlim(x0, x1)
	ax.set_ylim(x0, x1)
	ax.set_aspect("equal", adjustable="box")

	ax.set_title(title)
	ax.set_xlabel("Actual Temperature (degC)")
	ax.set_ylabel("Predicted Temperature (degC)")
	ax.grid(alpha=0.3)
	ax.legend(fontsize=9)


def print_metrics(name, metrics):
	print(f"\n[{name}]")
	print(f"MSE = {metrics['mse']:.4f}")
	print(f"MAE = {metrics['mae']:.4f}")
	print(f"Max single-point deviation = {metrics['max_dev']:.4f}")
	print(f"Deviation percentage = {metrics['max_dev_pct']:.2f}%")


if __name__ == "__main__":
	params_0 = build_parameter_set(0)
	params_1 = build_parameter_set(1)

	pred_ov_0, pred_non_0 = run_simulation_with_params(params_0)
	pred_ov_1, pred_non_1 = run_simulation_with_params(params_1)

	metrics_0 = calc_metrics(pred_ov_0, pred_non_0)
	metrics_1 = calc_metrics(pred_ov_1, pred_non_1)

	print_metrics("parameter_change(0)", metrics_0)
	print_metrics("parameter_change(1)", metrics_1)

	fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
	plot_parity(axes[0], pred_ov_0, pred_non_0, "Parity Plot: parameter_change(0)", metrics_0)
	plot_parity(axes[1], pred_ov_1, pred_non_1, "Parity Plot: parameter_change(1)", metrics_1)
	plt.show()