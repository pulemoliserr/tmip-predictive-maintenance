"""Cost-based evaluation and business-case (ROI / NPV / IRR / break-even) calculations."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


def confusion_at(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics_at(y, p, thr, c_fp=100, c_fn=400) -> dict:
    tp, fp, fn, tn = confusion_at(y, p, thr)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(threshold=thr, TP=tp, FP=fp, FN=fn, TN=tn,
                accuracy=(tp + tn) / max(1, tp + fp + fn + tn), sensitivity=rec,
                specificity=tn / (tn + fp) if tn + fp else 0.0, precision=prec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                cost=c_fp * fp + c_fn * fn)


def threshold_sweep(y, p, c_fp=100, c_fn=400, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    return pd.DataFrame([metrics_at(y, p, t, c_fp, c_fn) for t in grid])


@dataclass
class BusinessCase:
    machines: int = 150                      # monitored assets
    failures_per_machine_year: float = 0.8   # unplanned failures per asset and year today
    downtime_h_per_failure: float = 6.0
    cost_per_downtime_h: float = 800.0      # lost contribution margin per hour
    emergency_repair_cost: float = 3500.0    # parts, overtime, secondary damage
    planned_repair_cost: float = 2000.0      # repair when failure is caught early
    planned_downtime_h: float = 2.0
    inspection_cost: float = 350.0           # cost of one false alarm (technician visit)
    recall: float = 0.74
    precision: float = 0.86
    capex: float = 350_000.0                 # IoT retrofit, integration, project (year 1)
    opex_year: float = 90_000.0              # cloud, licences, 0.5 FTE data scientist, model maintenance
    adoption: tuple = (0.5, 0.85, 1.0, 1.0)  # ramp-up of usage per year
    discount_rate: float = 0.075
    years: int = 4

    def unit_economics(self) -> dict:
        unplanned = self.downtime_h_per_failure * self.cost_per_downtime_h + self.emergency_repair_cost
        planned = self.planned_downtime_h * self.cost_per_downtime_h + self.planned_repair_cost
        return dict(cost_unplanned=unplanned, cost_planned=planned, saving_per_catch=unplanned - planned)

    def yearly(self) -> pd.DataFrame:
        ue = self.unit_economics()
        failures = self.machines * self.failures_per_machine_year
        rows, cum = [], 0.0
        for yr in range(1, self.years + 1):
            a = self.adoption[min(yr - 1, len(self.adoption) - 1)]
            caught = failures * self.recall * a
            false_alarms = caught * (1 / max(self.precision, 1e-6) - 1)
            benefit = caught * ue["saving_per_catch"]
            fa_cost = false_alarms * self.inspection_cost
            cost = (self.capex if yr == 1 else 0.0) + self.opex_year + fa_cost
            net = benefit - cost
            cum += net
            rows.append(dict(year=yr, adoption=a, failures_caught=caught, false_alarms=false_alarms,
                             benefit=benefit, capex=self.capex if yr == 1 else 0.0, opex=self.opex_year,
                             false_alarm_cost=fa_cost, total_cost=cost, net=net, cumulative=cum,
                             npv_to_date=npv(self.discount_rate, [r["net"] for r in rows] + [net])))
        return pd.DataFrame(rows).set_index("year")

    def summary(self) -> dict:
        y = self.yearly()
        nets = y["net"].tolist()
        return dict(npv=npv(self.discount_rate, nets), irr=irr(nets), roi=y["net"].sum() / y["total_cost"].sum(),
                    payback_years=payback(nets), total_benefit=y["benefit"].sum(), total_cost=y["total_cost"].sum())

    def to_dict(self):
        return asdict(self)


def npv(rate, cashflows):
    """Cash flows at the end of years 1..n (as in the course's DCF example)."""
    return float(sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cashflows)))


def irr(cashflows, lo=-0.99, hi=10.0, tol=1e-7):
    f = lambda r: sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def payback(cashflows):
    """Amortisation time in years with linear interpolation inside the break-even year."""
    cum = 0.0
    for i, cf in enumerate(cashflows):
        if cum + cf >= 0 and cf > 0:
            return i + (-cum / cf if cum < 0 else 0.0)
        cum += cf
    return float("nan")


def sensitivity(bc: BusinessCase, params: list[str], delta=0.2) -> pd.DataFrame:
    """Tornado data: NPV when each parameter moves by ±delta."""
    base = bc.summary()["npv"]
    rows = []
    for p in params:
        out = {}
        for sgn in (-1, 1):
            d = bc.to_dict(); d[p] = d[p] * (1 + sgn * delta)
            if p in ("recall", "precision"):
                d[p] = min(d[p], 0.999)
            out[sgn] = BusinessCase(**d).summary()["npv"]
        rows.append(dict(parameter=p, low=out[-1] - base, high=out[1] - base, swing=abs(out[1] - out[-1])))
    return pd.DataFrame(rows).sort_values("swing"), base
