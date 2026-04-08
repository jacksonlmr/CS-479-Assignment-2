import sys
from rich.console import Console, Group
from rich.table import Table
from rich.rule import Rule
from rich import box

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
                title_style="bold white", show_lines=True)
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
    return make_legend("Missclassification Legend", [
        ("[bold white]█████[/]",        "Real parameter rate (baseline)"),
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