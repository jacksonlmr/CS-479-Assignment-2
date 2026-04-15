import sys
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
console = Console(width=220, legacy_windows=False)

def to_cpu(arr):
    return arr.get() if hasattr(arr, 'get') else arr

def fmt_rate(r):
    return f"{r*100:.4f}%"

def rich_delta(real, est):
    if real == 0:
        return "N/A"
    d = ((est - real) / real) * 100
    s = f"{d:+.2f}%"
    if abs(d) < 0.5:
        return f"[dim]{s}[/]"
    elif d > 5:
        return f"[#FF6666]{s}[/]"
    elif d > 0:
        return f"[dark_red]{s}[/]"
    elif d < -5:
        return f"[light_green]{s}[/]"
    else:
        return f"[dim light_green]{s}[/]"

def make_legend(title, entries):
    leg = Table(title=title, box=box.ROUNDED,
                title_style="bold white", show_lines=True,
                show_header=False)
    leg.add_column("Color", justify="center", no_wrap=True, min_width=7)
    leg.add_column("Meaning", justify="left")
    for swatch, meaning in entries:
        leg.add_row(swatch, meaning)
    return leg

def param_legend():
    return make_legend("Parameter Legend", [
        ("[bold yellow]█████[/]",   "True reference values"),
        ("[dim light_green]█████[/]",    "Mean (mu)"),
        ("[cyan]█████[/]",     "Diagonal sigma (11, 22)"),
        ("[magenta]█████[/]",  "Off-diagonal sigma (12, 21)"),
    ])

def rate_legend():
    return make_legend("Misclassification Legend", [
        ("[dim]█████[/]",          "Negligible change (+/- 0.5%)"),
        ("[dim light_green]█████[/]",        "Rate improved (< -0.5%)"),
        ("[light_green]█████[/]",   "Rate greatly improved (< -5%)"),
        ("[#FF6666]█████[/]",          "Rate worsened (> +0.5%)"),
        ("[dark_red]█████[/]",     "Rate greatly worsened (> +5%)"),
    ])

def with_legend(table, legend):
    grid = Table.grid(padding=(0, 3))
    grid.add_column()
    grid.add_column(vertical="middle")
    grid.add_row(table, legend)
    return grid

def build_param_table(title, cls, mu_true, sigma_true, est_params, fractions):
    t = Table(title=title, box=box.ROUNDED, show_lines=True,
              title_style="bold cyan", header_style="bold white")
    t.add_column("Sample",              style="bold white",  justify="center")
    t.add_column(f"mu{cls}_x",         style="light_green",       justify="right")
    t.add_column(f"mu{cls}_y",         style="light_green",       justify="right")
    t.add_column(f"sigma{cls}_11",     style="cyan",        justify="right")
    t.add_column(f"sigma{cls}_12",     style="magenta",     justify="right")
    t.add_column(f"sigma{cls}_21",     style="magenta",     justify="right")
    t.add_column(f"sigma{cls}_22",     style="cyan",        justify="right")

    mu = to_cpu(mu_true)
    s  = to_cpu(sigma_true)
    t.add_row(
        "[bold yellow]True[/]",
        f"[bold yellow]{mu[0]:.4f}[/]",  f"[bold yellow]{mu[1]:.4f}[/]",
        f"[bold yellow]{s[0,0]:.4f}[/]", f"[bold yellow]{s[0,1]:.4f}[/]",
        f"[bold yellow]{s[1,0]:.4f}[/]", f"[bold yellow]{s[1,1]:.4f}[/]",
    )
    for frac in reversed(fractions):
        mu_e, s_e = est_params[frac]
        mu_e = to_cpu(mu_e)
        s_e  = to_cpu(s_e)
        t.add_row(
            f"{frac*100:.4g}%",
            f"{mu_e[0]:.4f}", f"{mu_e[1]:.4f}",
            f"{s_e[0,0]:.4f}", f"{s_e[0,1]:.4f}", f"{s_e[1,0]:.4f}", f"{s_e[1,1]:.4f}",
        )
    return t

def build_rate_table(title, real_rate, idx, fractions, estimated_miss_rates, zeroed_miss_rates):
    t = Table(title=title, box=box.ROUNDED, show_lines=True,
              title_style="bold magenta", header_style="bold white")
    t.add_column("Sample Size",       style="bold cyan",  justify="center")
    t.add_column("Real Params",       style="bold white", justify="right")
    t.add_column("Estimated Params",  style="white",      justify="right")
    t.add_column("% Change (Est)",                        justify="right")
    t.add_column("Zeroed Diag",       style="white",      justify="right")
    t.add_column("% Change (Zeroed)",                     justify="right")
    for frac in fractions:
        e = estimated_miss_rates[frac]
        z = zeroed_miss_rates[frac]
        t.add_row(
            f"{frac*100:.4g}%",
            fmt_rate(real_rate),
            fmt_rate(e[idx]), rich_delta(real_rate, e[idx]),
            fmt_rate(z[idx]), rich_delta(real_rate, z[idx]),
        )
    return t


def plot_roc(t_values, bayesian_error_results, name):
    plt.figure()
    plt.plot(t_values, bayesian_error_results[:, 0], label='FPR')                                                                                                                                                                                                              
    plt.plot(t_values, bayesian_error_results[:, 1], label='FNR')    
    fpr_vals = bayesian_error_results[:, 0]                                                                                                                                                                                                                                    
    fnr_vals = bayesian_error_results[:, 1]                                                                                                                                                                                                                                    
    
    # Find where FPR - FNR changes sign (crossing point)                                                                                                                                                                                                                       
    diff = fpr_vals - fnr_vals                                                                                                                                                                                                                                               
    idx = np.where(np.diff(np.sign(diff)))[0][0]

    # Linear interpolation for exact t and rate at intersection
    t0, t1 = t_values[idx], t_values[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    t_intersect = t0 - d0 * (t1 - t0) / (d1 - d0)
    rate_intersect = fpr_vals[idx] + (fpr_vals[idx + 1] - fpr_vals[idx]) * (t_intersect - t0) / (t1 - t0)

    plt.axvline(x=t_intersect, color='gray', linestyle='--')
    plt.annotate(
        f't = {t_intersect:.4f}\nrate = {rate_intersect:.4f}',
        xy=(t_intersect, rate_intersect),
        xytext=(t_intersect + (t_values[-1] - t_values[0]) * 0.05, rate_intersect),
        arrowprops=dict(arrowstyle='->'),
    )                                                                                                                                                                                                   
    plt.xlabel('Threshold t')                                                                                                                                                                                                                                                  
    plt.ylabel('Error Rate')                                                                                                                                                                                                                                                   
    plt.title(f'FPR and FNR vs Threshold ({name})')
    plt.legend()
    plt.savefig(f"{name}_roc.jpg")
    return t_intersect


def plot_roc_2(ber_a, ber_b, filename):
    """
    Plots ROC curves (FPR vs FNR) comparing part a and part b color spaces.
    """
    plt.figure()
    plt.plot(ber_a[:, 0], ber_a[:, 1], label='Part A (Chromatic)')
    plt.plot(ber_b[:, 0], ber_b[:, 1], label='Part B (YCbCr)')
    plt.ylim(0, 0.5)
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.title('ROC Curves: Part A vs Part B')
    plt.legend()
    plt.savefig(filename)