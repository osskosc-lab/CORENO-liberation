"""Create a compact, visually checked PDF report from frozen Phase-1 outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def p(text, style): return Paragraph(text, style)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--output", default="results/CORENO_MultiAgent_Phase1_Report.pdf"); args = ap.parse_args()
    root = Path(args.results); out = Path(args.output)
    with open(root / "decision.json", encoding="utf-8") as f: d = json.load(f)
    s = pd.read_csv(root / "condition_summary.csv")
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=8)); styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.3, leading=10.5)); styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=6))
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    story = [p("CORENO Multi-Agent Phase 1", styles["Title2"]), p("Falsification report - frozen synthetic shift experiment", styles["Heading3"]), Spacer(1, 5)]
    verdict_color = {"SUPPORTED":"#1e8449", "FALSIFIED":"#b03a2e", "INCONCLUSIVE":"#b9770e"}[d["verdict"]]
    story += [p(f"<b>Decision: <font color='{verdict_color}'>{d['verdict']}</font></b>", styles["Heading2"]), p("This report tests only a constrained engineering claim: whether a particular integration controller improves immediate adaptation to a deliberately adversarial correlation reversal. It is not evidence for CORENO ontology, consciousness, or liberation.", styles["Body2"])]
    qrows = [["Primary metric", "Value"], ["Q = CORENO / strong baseline", f"{d['q_coreno_over_strong']:.3f}"], ["Bootstrap 95% CI", f"[{d['q_ci95_low']:.3f}, {d['q_ci95_high']:.3f}]"], ["High-confidence wrong-rate delta", f"{d['highconf_wrong_rate_delta']:+.3f}"], ["CORENO STOP rate", f"{d['coreno_stop_rate']:.3f}"], ["Frozen primary window", "steps 1000-1299"]]
    t=Table(qrows,colWidths=[82*mm,76*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1f4e79")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#b7c9d6")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f6f9fb")),("PADDING",(0,0),(-1,-1),6)])); story += [t, Spacer(1,8)]
    story += [p("Protocol", styles["Heading2"]), p("Five fixed online logistic agents receive identical observations and feedback in every condition. In stable exposure, a spurious feature is 90% accurate; at unseen shifts it becomes 10% accurate while a causal feature stays 70% accurate. The strongest comparator has confidence weighting, Hedge, change detection, and costed abstention. CORENO adds bounded plural weights, explicit dissent, self-stop, and partial reconstruction.", styles["Body2"]), Spacer(1,4)]
    cols = [["Condition", "Primary loss", "95% seed CI", "STOP", "High-conf wrong"]]
    for _, r in s.iterrows(): cols.append([r["condition"].replace("coreno_","C3-"), f"{r.primary_loss_mean:.3f}", f"[{r.primary_loss_ci_low:.3f}, {r.primary_loss_ci_high:.3f}]", f"{r.stop_rate_mean:.3f}", f"{r.highconf_wrong_rate_mean:.3f}"])
    tab=Table(cols,colWidths=[40*mm,29*mm,42*mm,25*mm,38*mm], repeatRows=1); tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1f4e79")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#c8d6df")),("FONTSIZE",(0,0),(-1,-1),7.2),("PADDING",(0,0),(-1,-1),3),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f9fb")])]))
    story += [tab, PageBreak(), p("Ablations, null controls, and interpretation", styles["Heading2"]), Image(str(root / "primary_loss_by_condition.png"), width=172*mm, height=78*mm), Spacer(1,2), Image(str(root / "paired_primary_loss.png"), width=90*mm, height=80*mm), Spacer(1,2)]
    full = s.set_index("condition").loc["coreno_full", "primary_loss_mean"]
    nulls = s[s.condition.str.startswith("coreno_") & (s.condition != "coreno_full")].copy(); nulls["delta"] = nulls.primary_loss_mean - full
    notes = [f"<b>{r.condition.replace('coreno_','C3-')}</b>: primary-loss change versus full = {r.delta:+.3f}." for _,r in nulls.iterrows()]
    story += [p("Explanatory checks", styles["Heading3"])] + [p(x, styles["Body2"]) for x in notes]
    if d["verdict"] == "SUPPORTED": conclusion = "The predeclared engineering claim survived this particular test. The component checks remain necessary before assigning the gain to any individual CORENO mechanism."
    elif d["verdict"] == "FALSIFIED": conclusion = "The primary claim is falsified under this frozen environment and comparator: CORENO did not beat the strong baseline. This is a result about this architecture and test, not a disproof of broader philosophical language."
    else: conclusion = "The experiment does not meet the strict 10% improvement criterion and does not demonstrate non-inferiority failure. The correct conclusion is inconclusive; further data must be a newly registered phase, not extra seeds added to this result."
    story += [p("Conclusion", styles["Heading2"]), p(conclusion, styles["Body2"]), p("Fixed stopping rule: SUPPORTED requires the Q upper CI <= 0.90 plus both safety gates. FALSIFIED requires Q lower CI >= 1.00. Otherwise the result is INCONCLUSIVE.", styles["Small"])]
    doc.build(story)

if __name__ == "__main__": main()
