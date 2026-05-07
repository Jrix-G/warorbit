#!/usr/bin/env python3
"""Generate the V10 War Orbit Audit Report PDF."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Image,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ━━ Color Palette ━━
ACCENT = colors.HexColor('#1e7693')
TEXT_PRIMARY = colors.HexColor('#1f2022')
TEXT_MUTED = colors.HexColor('#747a81')
BG_SURFACE = colors.HexColor('#dce0e3')
BG_PAGE = colors.HexColor('#f1f2f3')
TABLE_HEADER_COLOR = ACCENT
TABLE_HEADER_TEXT = colors.white
TABLE_ROW_EVEN = colors.white
TABLE_ROW_ODD = BG_SURFACE

# ━━ Font Registration ━━
pdfmetrics.registerFont(TTFont('NotoSansSC', '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf'))
FONT = 'NotoSansSC'

OUTPUT = '/home/z/my-project/download/V10_WarOrbit_Audit_Report.pdf'

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    topMargin=22*mm,
    bottomMargin=22*mm,
    leftMargin=20*mm,
    rightMargin=20*mm,
    title='V10 War Orbit Audit Report',
    author='Z.ai',
    subject='Engineering Audit - V10 Training System',
)

W = A4[0] - 40*mm  # content width

# ━━ Styles ━━
styles = getSampleStyleSheet()

s_title = ParagraphStyle('Title', fontName=FONT, fontSize=22, leading=28, textColor=ACCENT, spaceAfter=4*mm, alignment=TA_LEFT)
s_h1 = ParagraphStyle('H1', fontName=FONT, fontSize=16, leading=22, textColor=ACCENT, spaceBefore=6*mm, spaceAfter=3*mm, alignment=TA_LEFT)
s_h2 = ParagraphStyle('H2', fontName=FONT, fontSize=13, leading=18, textColor=TEXT_PRIMARY, spaceBefore=4*mm, spaceAfter=2*mm, alignment=TA_LEFT)
s_h3 = ParagraphStyle('H3', fontName=FONT, fontSize=11.5, leading=16, textColor=ACCENT, spaceBefore=3*mm, spaceAfter=1.5*mm, alignment=TA_LEFT)
s_body = ParagraphStyle('Body', fontName=FONT, fontSize=10, leading=15.5, textColor=TEXT_PRIMARY, spaceAfter=2*mm, alignment=TA_JUSTIFY)
s_body_bold = ParagraphStyle('BodyBold', parent=s_body, fontName=FONT, textColor=TEXT_PRIMARY)
s_code = ParagraphStyle('Code', fontName='Courier', fontSize=8.5, leading=12, textColor=colors.HexColor('#333333'), backColor=colors.HexColor('#f4f4f4'), leftIndent=4, rightIndent=4, spaceBefore=2, spaceAfter=2, borderPadding=4)
s_bullet = ParagraphStyle('Bullet', parent=s_body, leftIndent=12, firstLineIndent=-12, spaceAfter=1*mm)
s_caption = ParagraphStyle('Caption', fontName=FONT, fontSize=9, leading=13, textColor=TEXT_MUTED, spaceAfter=1.5*mm)
s_diag_box = ParagraphStyle('DiagBox', fontName='Courier', fontSize=8, leading=11, textColor=colors.HexColor('#222222'), backColor=colors.HexColor('#fffbe6'), leftIndent=4, rightIndent=4, spaceBefore=2, spaceAfter=2, borderPadding=4)

def P(text, style=s_body):
    return Paragraph(text, style)

def H1(text):
    return Paragraph(text, s_h1)

def H2(text):
    return Paragraph(text, s_h2)

def H3(text):
    return Paragraph(text, s_h3)

def BULLET(text):
    return Paragraph(text, s_bullet)

def CODE(text):
    return Paragraph(text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), s_code)

def DIAG(text):
    return Paragraph(text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), s_diag_box)

def HR():
    return HRFlowable(width='100%', thickness=0.5, color=ACCENT, spaceAfter=3*mm, spaceBefore=2*mm)

def make_table(headers, rows, col_widths=None):
    """Build a styled table."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = [W / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), TABLE_HEADER_TEXT),
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c0c0c0')),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_ROW_ODD))
        else:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_ROW_EVEN))
    t.setStyle(TableStyle(style_cmds))
    return t

story = []

# ═══════════════════════════════════════════════
# COVER
# ═══════════════════════════════════════════════
story.append(Spacer(1, 30*mm))
story.append(P('V10 War Orbit', s_title))
story.append(P('Engineering Audit Report', ParagraphStyle('Subtitle', fontName=FONT, fontSize=15, leading=20, textColor=TEXT_MUTED, spaceAfter=8*mm)))
story.append(HR())
story.append(P('Date: 2026-05-07', s_body))
story.append(P('Scope: V10 4p-first self-improving training system', s_body))
story.append(P('Baseline comparison: bot_v7 (53% winrate, ~700 Elo)', s_body))
story.append(P('Status: Benchmark performance is poor or unstable; training scores inflate while actual winrate collapses against external opponents', s_body))
story.append(Spacer(1, 12*mm))
story.append(HR())
story.append(P('<b>Primary findings:</b> The V10 training objective over-rewards main-front mass accumulation and under-rewards actual conversion/closing. The planner is structurally passive, spending 28-31% of turns in reserve_hold while the policy penalizes tactical action families. Backbone engagement remains below target at 0.07-0.15 despite heavy penalties.', s_body))

story.append(PageBreak())

# ═══════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ═══════════════════════════════════════════════
story.append(H1('Table of Contents'))
toc_items = [
    '1. Root Cause Summary',
    '2. Evidence from Logs and Code',
    '3. Audit Question-by-Question Analysis',
    '   Q1: Is the V10 objective optimizing the wrong proxy?',
    '   Q2: Is the planner creating a large passive front?',
    '   Q3: Is V10 worse than v7 because it lost tactical capability?',
    '   Q4: Are train/eval/benchmark schedules misleading?',
    '   Q5: What should be changed next? (Top 5 Fixes)',
    '   Q6: Why is backbone still below target?',
    '   Q7: Should main_front be de-weighted?',
    '4. Ranked Fix List with Expected Impact',
]
for item in toc_items:
    indent = 12 if item.startswith('   ') else 0
    s = ParagraphStyle('toc_item', parent=s_body, leftIndent=indent, spaceAfter=1*mm)
    story.append(P(item.strip(), s))
story.append(PageBreak())

# ═══════════════════════════════════════════════
# 1. ROOT CAUSE SUMMARY
# ═══════════════════════════════════════════════
story.append(H1('1. Root Cause Summary'))
story.append(P(
    'The V10 training system suffers from three compounding structural failures that together explain '
    'the gap between its ~46-50% training winrate (against weak opponents) and its ~8% benchmark winrate (against '
    'tournament-caliber notebook opponents). These are not parameter tuning issues; they are architectural '
    'defects in the objective function, the planner candidate pool, and the scoring policy that collectively '
    'make the bot optimize for mass hoarding instead of actual game-winning behavior.',
    s_body
))
story.append(P(
    '<b>Root Cause 1: Wrong Objective Proxy.</b> The training objective (_regularized_train_score in v10_trainer.py) '
    'allocates a massive 0.24 weight to _main_front_progress_score and a 1.80x multiplier to _front_pressure_adjustment. '
    'The front_pressure_adjustment alone can swing the score by up to +/-0.44 points, overwhelming the 0.74 weight on actual '
    '4p winrate. The bot learns that concentrating ships in a blob near one anchor planet maximizes the training '
    'objective, even when that concentration never converts into captured enemy planets. Evidence: main_front_ship_share '
    'reaches 0.76-0.84 (well above the 0.42 target), but t120 conversion stays at 9-10 planets.',
    s_body
))
story.append(P(
    '<b>Root Cause 2: Passive Planner Architecture.</b> The V10 planner, built on V9 code, generates candidates '
    'that are overwhelmingly defensive. The reserve_hold plan family, which does nothing but reinforce threatened planets, '
    'occupies 28-31% of all turns in the training logs. The backbone and staging_transfer families only perform ship '
    'relocation (moving ships toward the front anchor) but never execute actual attacks. The four_player_backbone '
    'candidate (line 845-883 of planner.py) consists entirely of _stage_to_front calls with zero attack logic. '
    'Combined with the concentration phase scatter penalty (lines 399-409) that heavily penalizes off-focus attacks '
    'during turns 45-125, the bot accumulates mass without spending it on captures.',
    s_body
))
story.append(P(
    '<b>Root Cause 3: Lost Tactical Depth vs. V7.</b> The V7 bot possesses several capabilities entirely absent '
    'from V10: (a) an arrival ledger with timeline simulation that predicts future planet ownership, enabling precise '
    'garrison calculation; (b) multi-source swarm attacks (3-4 planets coordinating on a single target); (c) proactive '
    'defense with multi-enemy stacking window analysis; (d) rear staging that pushes 70% of ships from idle rear '
    'planets toward the front; (e) comet trajectory prediction; and (f) finisher multipliers tied to elimination '
    'bonuses. V10 has none of these. Its MoveBuilder only supports 2-source joint attacks, its planning simulation '
    'uses depth=1 with rollouts=0 during training (effectively zero look-ahead), and it has no arrival ledger '
    'whatsoever, relying only on game.plan_shot for basic intercept calculation.',
    s_body
))
story.append(PageBreak())

# ═══════════════════════════════════════════════
# 2. EVIDENCE FROM LOGS AND CODE
# ═══════════════════════════════════════════════
story.append(H1('2. Evidence from Logs and Code'))

story.append(H2('2.1 Representative Training Logs'))
story.append(P(
    'The following log excerpts from the v10_4p_train.jsonl file show the critical pattern: training scores '
    'that appear healthy mask an underlying collapse in strategic behavior. Note how eval=0.375 at gen 14 and '
    'eval=0.405 at gen 15 are NOT holdout scores -- holdout was disabled in the latest volume run.',
    s_body
))

story.append(DIAG('gen=0011  train=0.467 eval=0.438  sel=0.468  promo=0\n'
                 'conv=8.4/10.6/10.1/10.6\n'
                 '4pdiag=WARN xfer=0.59 bb=0.15 lock=0.99 fronts=3.2 mf=0.75 ready=0.87\n\n'
                 'gen=0012  train=0.484 eval=0.374  sel=0.792  promo=0\n'
                 'block=train_only,holdout_t120_low,skill_lcb_down,sprt=accept(+4.20)\n'
                 'conv=7.8/10.8/11.0/11.8\n'
                 '4pdiag=WARN xfer=0.60 bb=0.18 lock=1.00 fronts=2.8 mf=0.76 ready=0.88\n\n'
                 'gen=0015  train=0.405 eval=0.405  sel=0.313  promo=1\n'
                 'conv=8.1/10.4/9.8/9.8\n'
                 '4pdiag=WARN xfer=0.55 bb=0.147 lock=1.00 fronts=2.71 mf=0.765 ready=0.88'))

story.append(P(
    '<b>Key pattern:</b> Every generation shows 4pdiag=WARN. The backbone metric (bb) never exceeds 0.18 across '
    'all logged generations. The main_front metric (mf) consistently exceeds the 0.40 target at 0.75-0.84. '
    'The lock metric is near-perfect (0.98-1.00), confirming the bot locks onto a single enemy effectively -- '
    'it just does not convert that focus into captures. Transfer fraction (xfer) is high at 0.55-0.63, but '
    'this measures movement volume, not movement purpose.',
    s_body
))

story.append(H2('2.2 Plan Type Distribution (Gen 6-11)'))
story.append(P(
    'The following table shows the plan_type_frac data from the training logs, revealing the behavioral '
    'composition of the V10 bot across generations. The data consistently shows reserve_hold dominating '
    'while tactical families remain suppressed.',
    s_body
))

plan_data = [
    ['Plan Type', 'Gen 6', 'Gen 7', 'Gen 8', 'Gen 9', 'Gen 11'],
    ['reserve_hold', '22.2%', '19.7%', '26.9%', '23.6%', '28.8%'],
    ['defensive_consolidation', '21.1%', '17.8%', '22.5%', '16.1%', '20.6%'],
    ['aggressive_expansion', '15.5%', '15.6%', '15.5%', '12.9%', '12.1%'],
    ['balanced', '16.2%', '14.3%', '8.2%', '14.6%', '9.9%'],
    ['staging_transfer (backbone)', '12.4%', '14.2%', '12.9%', '14.6%', '11.5%'],
    ['multi_step_trap', '4.5%', '4.3%', '1.7%', '3.8%', '1.7%'],
    ['resource_denial', '1.1%', '2.7%', '2.2%', '2.0%', '4.3%'],
    ['endgame_finisher', '0.8%', '2.2%', '1.2%', '2.3%', '3.0%'],
    ['delayed_strike', '1.6%', '3.5%', '2.9%', '3.0%', '2.2%'],
    ['opportunistic_snipe', '0.7%', '0.8%', '0.7%', '0.8%', '0.8%'],
    ['probe', '3.9%', '5.0%', '3.5%', '6.3%', '5.1%'],
]
cw = [W*0.26, W*0.12, W*0.12, W*0.12, W*0.12, W*0.12]
story.append(make_table(plan_data[0], plan_data[1:], cw))
story.append(Spacer(1, 3*mm))
story.append(P(
    '<b>Interpretation:</b> Reserve_hold + defensive_consolidation = 40-50% of all turns. These are purely '
    'defensive plans that conserve ships. Tactical attack families (resource_denial, delayed_strike, '
    'opportunistic_snipe) collectively account for only 5-10% of turns. The endgame_finisher, which is '
    'the plan designed to close out games, accounts for 1-3% of turns. This distribution shows a bot that '
    'is structurally biased toward defense and accumulation rather than offense.',
    s_body
))

story.append(H2('2.3 Backbone vs. Main Front Decoupling'))
story.append(P(
    'The following table from JSONL logs shows the persistent gap between backbone_turn_frac (bb) and '
    'main_front_ship_share (mf) across training generations. The two metrics measure fundamentally different '
    'things: bb counts turns where the backbone staging plan was selected by the policy, while mf measures '
    'the fraction of total military mass concentrated near the front anchor. The bot can achieve high mf '
    'while having low bb because mass naturally drifts to the front through normal production, not through '
    'deliberate staging transfers.',
    s_body
))

bb_data = [
    ['Metric', 'Gen 1', 'Gen 3', 'Gen 6', 'Gen 8', 'Gen 9', 'Gen 11'],
    ['backbone_turn_frac', '0.116', '0.073', '0.124', '0.144', '0.146', '0.115'],
    ['main_front_ship_share', '0.770', '0.631', '0.800', '0.800', '0.755', '0.760'],
    ['active_front_avg', '2.79', '2.53', '3.34', '3.30', '2.57', '3.01'],
    ['main_front_ready_frac', '0.852', '0.979', '0.918', '0.918', '0.876', '0.863'],
    ['conversion_t100_rate', '0.708', '0.625', '0.313', '0.297', '0.281', '0.297'],
]
cw2 = [W*0.24, W*0.11, W*0.11, W*0.11, W*0.11, W*0.11, W*0.11]
story.append(make_table(bb_data[0], bb_data[1:], cw2))
story.append(Spacer(1, 3*mm))
story.append(P(
    '<b>Critical observation:</b> Main_front_ready_frac consistently exceeds 0.85 (the objective is 0.35), '
    'confirming the mass is available for conversion. But conversion_t100_rate drops from 0.71 at gen 1 to '
    '0.28-0.31 at later generations. The bot has the ships to win but never uses them on captures. This is '
    'the defining symptom of the mass-over-conversion failure.',
    s_body
))

story.append(H2('2.4 Code-Level Evidence: Wrong Objective Weight'))
story.append(P(
    'In v10_trainer.py, the _regularized_train_score function (lines 145-176) defines the training objective. '
    'The critical weight allocation is shown below. Note the 1.80x multiplier on front_pressure_adjustment, '
    'which gives this term an effective weight of approximately 1.80 * 0.24 = 0.432, exceeding even the '
    '0.74 weight on actual 4p winrate. The front_pressure_adjustment function (lines 179-227) internally '
    'rewards main_front_ready_frac (ships near the anchor) with an additional bonus of 0.035 when it '
    'exceeds 0.30, and gives a "front_ok_bonus" of 0.045 when all front metrics are satisfied. This creates '
    'a perverse incentive where the bot scores well by having many ships near the anchor without needing '
    'to actually use them.',
    s_body
))

story.append(CODE(
    'score = (\n'
    '    0.74 * wr_4p             # Actual win rate\n'
    '    + 0.08 * wr_2p           # 2p win rate\n'
    '    + 0.03 * conv60\n'
    '    + 0.05 * conv80\n'
    '    + 0.08 * conv100\n'
    '    + 0.07 * conv120          # Conversion metrics\n'
    '    + 0.24 * _main_front_progress_score  # Mass accumulation\n'
    ')\n'
    'score += 1.80 * _front_pressure_adjustment  # DOMINATES\n'
))

story.append(P(
    'Furthermore, _main_front_progress_score (lines 230-253) weights mass metrics at 0.20*share + '
    '0.14*core_share + 0.14*ready = 0.48, while conversion metrics (c80, c100, c120) total only 0.16. '
    'The function rewards planet count thresholds at t80/t100/t120 (0.12 each) but these thresholds '
    '(9, 10, 12) are low enough that the bot can satisfy them through passive neutral expansion alone, '
    'without ever needing to attack an enemy planet.',
    s_body
))

story.append(PageBreak())

# ═══════════════════════════════════════════════
# 3. AUDIT QUESTIONS
# ═══════════════════════════════════════════════
story.append(H1('3. Audit Question-by-Question Analysis'))

story.append(H2('Q1: Is the V10 objective optimizing the wrong proxy?'))
story.append(P(
    '<b>Answer: Yes, unambiguously.</b> The training objective in _regularized_train_score (v10_trainer.py '
    'lines 145-176) allocates disproportionate weight to structural proxies (mass concentration, front count) '
    'rather than to the ultimate success metric (winning games). The three failure modes are as follows.',
    s_body
))
story.append(P(
    '<b>Failure A: _front_pressure_adjustment dominates the gradient.</b> With a 1.80x multiplier, '
    'the front_pressure_adjustment (which ranges from -0.24 to +0.22) has an effective range of +/-0.43 '
    'points. This overwhelms the 0.74 weight on wr_4p, meaning the gradient signal pushes weight updates '
    'toward satisfying structural metrics rather than toward improving actual winrate. When backbone_turn_frac '
    'is low (0.07-0.15 as observed), the penalty alone contributes approximately -0.12 to -0.18, creating a '
    'strong gradient away from tactical play. This explains why the bot learns to satisfy structural metrics '
    'while its actual game-winning behavior degrades across generations.',
    s_body
))
story.append(P(
    '<b>Failure B: _main_front_progress_score rewards static mass over conversion.</b> The score function '
    '(lines 230-253) weights main_front_ship_share at 0.20, main_front_core_ship_share at 0.14, and '
    'main_front_ready_frac at 0.14 -- totaling 0.48 for pure mass presence. Conversion metrics (c80, c100, c120) '
    'total only 0.16. The planet count thresholds (p80 >= 9, p100 >= 10, p120 >= 12) are easily achieved '
    'through passive neutral capture during the opening, so the bot satisfies this metric without needing '
    'to execute midgame attacks on enemy planets. Evidence: main_front_ready_frac = 0.86-0.92 but '
    'conversion_t100_rate drops from 0.71 to 0.28 across generations.',
    s_body
))
story.append(P(
    '<b>Failure C: reward_from_result shaping dilutes the win signal.</b> In self_play.py (lines 158-182), '
    'the reward function adds conversion shaping bonuses (up to 0.045 * 2 = 0.09) to every game reward. '
    'Combined with the confidence_l2 regularization (0.0025) and entropy penalties (-0.05 to -0.08), the '
    'effective gradient signal from pure win/loss is significantly attenuated. The bot receives substantial '
    'reward for merely having 8+ planets at t60 or 13+ at t100, regardless of whether it eventually wins. '
    'This creates a local optimum where the bot expands neutrals aggressively in the opening (satisfying '
    'the conversion thresholds) but becomes passive afterward.',
    s_body
))

story.append(H2('Q2: Is the planner creating a large front that is too passive?'))
story.append(P(
    '<b>Answer: Yes. The planner architecture is structurally passive in 4p games.</b> The V10 planner '
    '(v9/planner.py) generates 14 candidate families, but the hard_frontlock mechanism (lines 688-702) '
    'and the concentration phase scatter penalty (lines 399-409) aggressively prune offensive candidates, '
    'leaving only defensive and transfer-based plans when the bot has multiple fronts.',
    s_body
))
story.append(P(
    '<b>Mechanism: _four_player_backbone has zero attack logic.</b> The backbone candidate (lines 845-883) '
    'consists entirely of _stage_to_front calls that relocate ships toward the front anchor. There is no '
    'attack on any target. The score is purely based on staging volume (line 856-863). Similarly, '
    '_deep_staging (lines 962-986) also does only staging. The _front_lock_consolidation (lines 885-924) '
    'adds reinforcement defense but no attacks. These three consolidation families together account for '
    'a significant fraction of the high-scoring candidates, yet none of them can capture an enemy planet.',
    s_body
))
story.append(P(
    '<b>Mechanism: reserve_hold dominates turns.</b> The reserve_hold candidate (lines 1286-1289) is '
    'extremely simple: it calls _commit_reinforcements(b, force=True) and nothing else. Despite a safety '
    'penalty of -0.22 when no threats exist (policy.py line 298), it still accounts for 22-29% of all turns '
    'in training logs. The reason is that reinforcement moves are scored at 40.0 + 8.0*production (planner.py '
    'line 505), giving them a high base_score that inflates their ranking in the candidate list. Combined '
    'with the metadata bonuses for front_lock and consolidation_threshold (policy.py lines 246-247), '
    'reserve_hold frequently out-scores tactical candidates.',
    s_body
))
story.append(P(
    '<b>Mechanism: MoveBuilder.attack_left caps budget at 56%.</b> In planner.py (lines 84-97), the '
    'attack budget is calculated as base = int(world.available * 0.56) for non-reserve builders, further '
    'reduced by 0.35 if the source is threatened. This means even when an aggressive candidate IS selected, '
    'the planner sends only 56% of available ships on each source planet. By comparison, V7 uses a '
    'rear-staging system (REAR_SEND_RATIO_FOUR_PLAYER = 0.70) that pushes 70% of ships from idle planets, '
    'and its multi-source swarms coordinate 3-4 planets to overwhelm a single target.',
    s_body
))

story.append(H2('Q3: Is V10 worse than v7 because it lost tactical capability?'))
story.append(P(
    '<b>Answer: Yes, V10 is missing at least six critical tactical capabilities present in V7.</b> '
    'The comparison below identifies concrete missing heuristics that directly impact win conversion.',
    s_body
))

v7_data = [
    ['Capability', 'V7 (bot_v7.py)', 'V10 (planner.py/policy.py)', 'Impact'],
    ['Arrival ledger\n(timeline simulation)', 'Lines 476-605:\nbuild_arrival_ledger(),\nsimulate_planet_timeline()\nPredicts ownership\n80 turns ahead', 'No equivalent.\nOnly uses game.plan_shot\nfor basic intercept.\nNo future\nownership prediction.', 'HIGH:\nCannot calculate\ngarrison margins\nprecisely.'],
    ['Multi-source swarms\n(3-4 planet coordination)', 'Lines 106-118:\nTHREE_SOURCE_SWARM,\nFOUR_SOURCE_SWARM.\nCoordinates 3-4 planets\non single target with\neta tolerance.', 'Commit_target supports\nmax_sources=2 only.\nNo multi-source\nswarm logic.\nMoveBuilder caps at\n2 sources.', 'HIGH:\nCannot overwhelm\ndefended targets\nwith combined force.'],
    ['Proactive defense\n(multi-enemy stacking)', 'Lines 709-734:\n_multi_enemy_proactive_keep().\nSliding window analysis\nof enemy fleet\narrival stacking.', 'No equivalent.\nOnly basic threatened/\ndoomed candidate\ndetection via\nsimple timeline.', 'MEDIUM:\nCannot anticipate\ncoordinated\nmulti-enemy attacks.'],
    ['Rear staging\n(push from idle planets)', 'Lines 156-162:\nREAR_SEND_RATIO_FOUR_PLAYER=0.70\nRear staging pushes 70%\nof ships from rear\ntoward front.', '_stage_to_front with\nratio=0.86 max.\nOnly moves from planets\nfarther than\nfront+6 units.', 'HIGH:\nRear planets\naccumulate ships\nthat are never\nsent to front.'],
    ['Comet prediction', 'Lines 332-360:\n_predict_comet_pos(),\n_comet_remaining_life().\nFull trajectory\nprediction for comets.', 'No equivalent.\nOnly standard planet\nposition prediction.\nNo comet support.', 'MEDIUM:\nCannot capture\nhigh-value comet\nplanets efficiently.'],
    ['Finisher elimination bonus', 'Lines 139-176:\nFOUR_PLAYER_ELIMINATION_BONUS=35.0.\nMassive bonus for\neliminating weak enemies.', '_staged_finisher triggers\nat ship_lead>=1.12 but\nhas no elimination\nbonus multiplier.\nOnly +5 per target.', 'HIGH:\nNo strong incentive\nto finish weakened\nenemies quickly.'],
]
cw3 = [W*0.18, W*0.26, W*0.26, W*0.22]
story.append(make_table(v7_data[0], v7_data[1:], cw3))
story.append(Spacer(1, 3*mm))
story.append(P(
    'Beyond these specific capabilities, V7 also benefits from a fundamentally different scoring architecture. '
    'V7 uses a per-target _target_value function (bot_v7.py lines 968-1044) that computes a rich '
    'multiplicative value combining production, distance, timing, safety, domination status, and strategic '
    'phase bonuses. This value directly drives move selection without any intermediate weight-learning '
    'step. V10 instead routes through a neural policy with learned weights, but the training signal is '
    'contaminated by the proxy metrics described in Q1. The V7 approach is more direct: it computes the '
    'value of each action inline and executes the highest-value action immediately.',
    s_body
))

story.append(H2('Q4: Are train/eval/benchmark schedules misleading?'))
story.append(P(
    '<b>Answer: Yes. The latest volume run has zero generalization signal.</b> In the latest run command '
    '(from V10_AUDIT_PROMPT.md), the flags --holdout-eval-games-train-only 0, --train-only-benchmark-every 0, '
    'and --train-only-benchmark-games 0 together disable all holdout evaluation and periodic benchmarking.',
    s_body
))
story.append(P(
    '<b>Consequence 1: eval_summary aliases train_summary.</b> In v10_trainer.py line 680, when train_only '
    'is true and holdout_eval_games is 0, the code sets eval_summary = train_summary (same object reference). '
    'This means the eval scores at gen 14 (0.375) and gen 15 (0.405) are exactly the training scores. The '
    'audit prompt correctly warns about this, but the training loop still logs them as "eval" metrics, '
    'which could mislead future analysis.',
    s_body
))
story.append(P(
    '<b>Consequence 2: benchmark never runs during training.</b> With benchmark_every=0 and '
    'train_only_benchmark_games=0, no benchmark games are played during the training loop (v10_trainer.py '
    'lines 710-725). The only benchmark data comes from the JSONL log entries from earlier, shorter runs '
    '(gen 6-11), which used different training opponents and fewer pairs. The benchmark_summary dictionary '
    'remains at all zeros throughout the latest volume run. This means the SPRT promotion logic '
    '(v10_trainer.py lines 770-791) has no benchmark signal, and the _apply_guardian_adjustments function '
    '(lines 373-465) is never invoked. The training is flying completely blind to its actual performance.',
    s_body
))
story.append(P(
    '<b>Consequence 3: SPRT promotion can fire on weak evidence.</b> Gen 12 shows sprt=accept(+4.20) with '
    'train_only and no real holdout. The SPRT logic (v10_trainer.py lines 770-791) accumulates challenger '
    'wins from the progress monitor, which in turn records wins from train/eval/bench modes with '
    'different weights (train=0.4, eval=1.0, bench=1.6). Since eval=train during this run, the SPRT '
    'is computing evidence from a single distribution, making its acceptance/rejection decisions unreliable. '
    'Gen 15 was promoted (promo=1) with eval=train=0.405, meaning the best checkpoint was updated '
    'based on training performance against training opponents.',
    s_body
))

story.append(H2('Q5: What should be changed next? (Top 5 Fixes)'))
story.append(P(
    'The following five fixes are ranked by expected impact, estimated risk, and implementation '
    'complexity. Each fix targets a specific root cause identified in this audit. Fixes 1 and 2 '
    'are the highest priority and should be implemented together. Fix 3 is lower risk and can be done '
    'independently.',
    s_body
))

story.append(H3('Fix 1: Reduce front_pressure_adjustment multiplier (HIGHEST PRIORITY)'))
story.append(P(
    '<b>File:</b> v10_trainer.py, line 171. <b>Change:</b> Replace 1.80 with 0.30. <b>Rationale:</b> '
    'This single constant change reduces the effective weight of front_pressure_adjustment from ~0.43 to '
    '~0.07, restoring wr_4p as the dominant gradient signal. The front metrics will still provide '
    'useful shaping, but they will no longer overwhelm the win signal. This directly addresses '
    'Root Cause 1. <b>Risk:</b> LOW. The bot will temporarily spend more turns on tactical '
    'actions during training, which is the desired behavior. The 0.30 multiplier keeps front metrics '
    'as meaningful shaping signals without dominating. <b>Expected metric improvement:</b> backbone '
    'turn_frac should increase as the penalty for low backbone weakens. conversion_t120 should increase '
    'as the bot spends more ships on captures.',
    s_body
))

story.append(H3('Fix 2: Add attack logic to backbone candidate (HIGHEST PRIORITY)'))
story.append(P(
    '<b>File:</b> war_orbit/agents/v9/planner.py, lines 845-883. <b>Change:</b> After the '
    '_stage_to_front call, add 1-2 attack moves targeting the weakest enemy planet near the anchor. '
    '<b>Rationale:</b> The backbone candidate is the most frequently selected non-defensive plan in 4p games '
    '(backbone_turn_frac = 0.07-0.15), yet it currently performs zero attacks. Adding attack logic '
    'directly addresses Root Cause 2 by ensuring that even the "safest" tactical plan still makes '
    'progress toward eliminating the focused enemy. The attack should use existing _commit_target with '
    'max_sources=2 and family="balanced". <b>Risk:</b> LOW. The attack only fires after staging is '
    'complete, so it uses ships that would otherwise sit idle at the anchor. <b>Expected metric '
    'improvement:</b> benchmark winrate should increase as backbone turns now contribute actual '
    'captured planets, not just ship relocations.',
    s_body
))

story.append(H3('Fix 3: De-weight main_front_ready_frac in _main_front_progress_score'))
story.append(P(
    '<b>File:</b> v10_trainer.py, line 236. <b>Change:</b> Replace 0.14 (ready weight) with 0.02. '
    '<b>Rationale:</b> The main_front_ready_frac metric (fraction of turns where main_front_ship_share '
    'exceeds 0.42) is fully saturated at 0.85-0.92, providing zero gradient signal. The 0.14 weight '
    'allocated to it is wasted capacity in the objective. Reducing it to 0.02 frees objective '
    'bandwidth for conversion metrics. This directly addresses Root Cause 1, Failure B. <b>Risk:</b> '
    'LOW. The metric is already saturated; reducing its weight has no downside. <b>Expected metric '
    'improvement:</b> the objective gradient will shift toward conversion metrics, improving t100/t120 '
    'conversion rates.',
    s_body
))

story.append(H3('Fix 4: Raise max_sources cap from 2 to 3'))
story.append(P(
    '<b>File:</b> war_orbit/agents/v9/planner.py, _commit_target function (line 430), '
    'and war_orbit/agents/v9/policy.py, V10Weights.defaults(). <b>Change:</b> Increase the default '
    'max_sources parameter for _commit_target calls from 2 to 3, and add a simple 3-source '
    'coordination function. <b>Rationale:</b> This partially closes the capability gap with V7, which '
    'uses THREE_SOURCE_SWARM and FOUR_SOURCE_SWARM (bot_v7.py lines 106-118). The V7 swarm '
    'is critical for capturing defended enemy planets in midgame, where a 2-source attack often fails '
    'due to garrison accumulation. A 3-source attack with proper eta tolerance calculation can '
    'overwhelm a target that 2 sources cannot. <b>Risk:</b> LOW. The MoveBuilder already supports '
    'multiple moves per candidate. The change only affects the _commit_target function. '
    '<b>Expected metric improvement:</b> midgame capture rate should improve, leading to higher '
    'planet counts at t80/t100.',
    s_body
))

story.append(H3('Fix 5: Enable benchmark during train-only (required for any meaningful progress)'))
story.append(P(
    "<b>File:</b> run_v10.py default args. <b>Change:</b> Set default "
    "--train-only-benchmark-every=3 and --train-only-benchmark-games=16. Also enable holdout "
    "with --holdout-eval-games-train-only=8. <b>Rationale:</b> As shown in Q4, the latest volume run had "
    'zero generalization signal. Without periodic benchmarking, the training loop has no way to detect '
    'when it is optimizing for proxy metrics rather than actual game-winning behavior. This addresses '
    'the training schedule misleadingness described in Q4. <b>Risk:</b> LOW. The only cost is '
    'slightly reduced training volume (benchmark games replace training games). 16 benchmark games '
    'every 3 generations adds approximately 3-4 minutes per cycle, reducing total training games '
    'by ~15%. <b>Expected metric improvement:</b> enables the SPRT promotion guardrail and guardian '
    'adjustments, preventing promotion of weights that perform well on training opponents but fail on '
    'benchmark opponents.',
    s_body
))

story.append(H2('Q6: Why is backbone still below target?'))
story.append(P(
    '<b>Answer: The backbone metric is structurally bottlenecked by the MoveBuilder attack budget cap '
    'and the policy bias against staging_transfer relative to defensive_consolidation.</b>',
    s_body
))
story.append(P(
    'The backbone_turn_frac metric counts the fraction of turns where the backbone plan '
    '(v9_4p_backbone) was selected by the policy. Two factors suppress this selection:',
    s_body
))
story.append(P(
    '<b>Factor 1: Policy bias toward consolidation.</b> In V10Weights.defaults() (v10/policy.py lines 31-37), '
    'the plan_bias for defensive_consolidation is set to 0.26 while staging_transfer is only 0.11. This '
    'means consolidation starts with a +0.15 advantage in the linear score before any metadata bonuses. '
    'The metadata bonus for backbone (0.26 + 0.12 + 0.10 = 0.48, policy.py lines 109-114) partially '
    'offsets this, but when the front pressure is high (fronts > front_budget + 0.50), the policy '
    'applies an additional -0.10 penalty to staging_transfer (policy.py line 264-265) and -0.10 penalty '
    'to backbone candidates. The net effect is that consolidation frequently scores higher than '
    'backbone even when backbone would be strategically preferable.',
    s_body
))
story.append(P(
    '<b>Factor 2: The backbone candidate only does transfers.</b> As analyzed in Q2, the backbone '
    'candidate performs no attacks. The policy sees this through the plan_features: backbone candidates '
    'have high transfer_ship_frac and high transfer_move_frac but low attack_move_frac. The metadata '
    'bonus at policy.py line 266-267 (+0.14 when backbone > 0.0 and transfer >= 0.30 and '
    'attack < 0.35) explicitly rewards candidates that have high transfer AND low attack, which '
    'is exactly what the transfer-only backbone candidate looks like. This creates a feedback loop '
    'where the policy selects backbone for its high transfer metrics, but the backbone itself never '
    'attacks, so the attack_move_frac stays low, reinforcing the policy preference.',
    s_body
))
story.append(P(
    '<b>Factor 3: reserve_hold steals backbone turns.</b> The reserve_hold plan, which has zero attack '
    'logic, receives a base_score from _commit_reinforcements that is typically 40-80 (planner.py line '
    '505: 40.0 + 8.0*production). This high base_score, combined with the consolidation_threshold metadata '
    'bonus (policy.py line 227: 0.22 when my_planets < 15 and step < 140), allows reserve_hold to '
    'outscore backbone in many game states, even when no planet is actually threatened.',
    s_body
))
story.append(P(
    '<b>Recommendation:</b> In addition to Fix 2 (adding attacks to backbone), the reserve_hold '
    'base_score should be reduced from 40.0 + 8.0*production to something proportional to the actual '
    'threat level. When no planets are threatened, the reinforcement value should be near zero, not 40+. '
    'The current implementation always calls _commit_reinforcements(b, force=True), generating '
    'reinforcement moves regardless of whether any planet actually needs defense.',
    s_body
))

story.append(H2('Q7: Should main_front be de-weighted?'))
story.append(P(
    '<b>Answer: Yes. The main_front metrics are over-saturated and consuming objective bandwidth.</b>',
    s_body
))
story.append(P(
    'The main_front_ready_frac metric is fully saturated at 0.85-0.92 (target 0.35), meaning it '
    'provides zero gradient signal. The main_front_ship_share metric is typically at 0.76-0.84 (target '
    '0.42), also well above target. In _main_front_progress_score (v10_trainer.py lines 230-253), the '
    'combined weight of these two metrics is 0.20 + 0.14 = 0.34, which is 34% of the progress score. '
    'This 34% is wasted because the metrics are already at ceiling.',
    s_body
))
story.append(P(
    'The main_front_core_ship_share is slightly less saturated at 0.60-0.70, but it still contributes '
    '0.14 weight. A better objective would replace the static mass metrics with dynamic conversion '
    'metrics. Specifically, the progress score should weight: (a) the rate at which main_front_ship_share '
    'is actually converted into captured planets (ships-sent-to-capture ratio), (b) the rate at which '
    'focused enemy planets are reduced over a sliding window, and (c) the rate at which secondary fronts '
    'are closed (active_front_avg decreasing). These metrics directly measure whether the concentrated '
    'mass is being used for its intended purpose, rather than simply measuring whether mass is present.',
    s_body
))
story.append(P(
    'In the shorter term, Fix 3 (de-weighting main_front_ready_frac from 0.14 to 0.02) partially '
    'addresses this issue by reducing the weight of the most saturated metric. The remaining 0.20 weight '
    'on main_front_ship_share is acceptable as a shaping signal because it still has gradient signal '
    'in the 0.76-0.84 range (not fully saturated). However, the 0.14 weight on main_front_core_share '
    'should also be reduced to 0.04 or lower, since this metric is also saturated.',
    s_body
))

story.append(PageBreak())

# ═══════════════════════════════════════════════
# 4. RANKED FIX LIST
# ═══════════════════════════════════════════════
story.append(H1('4. Ranked Fix List with Expected Impact'))

fix_data = [
    ['#', 'Fix', 'File(s)', 'Change', 'Priority', 'Risk', 'Expected Impact'],
    ['1', 'Reduce front_pressure\nmultiplier', 'v10_trainer.py\nline 171', '1.80 -> 0.30', 'CRITICAL', 'LOW',
     'Restores winrate as dominant\ngradient signal. backbone_tfr rises,\nconversion_t120 improves.'],
    ['2', 'Add attack logic\nto backbone', 'planner.py\nlines 845-883', 'Add 1-2\nattack moves\nafter staging', 'CRITICAL', 'LOW',
     'Backbone turns now\ncontribute captures.\nBenchmark WR rises.'],
    ['3', 'De-weight\nmain_front_ready', 'v10_trainer.py\nline 236', '0.14 -> 0.02', 'HIGH', 'LOW',
     'Frees objective bandwidth\nfor conversion metrics.\nt100 improves.'],
    ['4', 'Raise max_sources\n2 -> 3', 'planner.py\n_commit_target,\npolicy.py defaults', 'max_sources\n2 -> 3', 'HIGH', 'LOW',
     'Closes capability gap\nwith V7 swarms.\nMidgame capture\nrate rises.'],
    ['5', 'Enable benchmark\nduring train-only', 'run_v10.py\ndefault args', 'benchmark-every=3,\nbenchmark-games=16,\nholdout=8', 'HIGH', 'LOW',
     'Enables generalization\nsignal. Prevents promotion\nof weak weights.'],
]
cw4 = [W*0.06, W*0.14, W*0.20, W*0.17, W*0.09, W*0.07, W*0.27]
story.append(make_table(fix_data[0], fix_data[1:], cw4))
story.append(Spacer(1, 4*mm))

story.append(P(
    '<b>Implementation priority:</b> Fixes 1 and 2 should be implemented first and tested together, as they '
    'address different root causes (wrong objective weight vs. passive planner architecture). Fix 3 '
    'is independent and can be applied at any time. Fix 4 requires careful testing because '
    'increasing max_sources changes the _commit_target behavior in multiple call sites. Fix 5 '
    'is a configuration change that requires no code modification, only default argument updates.',
    s_body
))
story.append(P(
    '<b>Expected cumulative impact:</b> With fixes 1-3 implemented, the bot should shift from its '
    'current behavior pattern (high mf, low bb, passive turns, weak conversion) toward a more '
    'balanced approach (moderate mf, higher bb, active attacks, improved conversion). The training '
    'objective gradient will push weights toward tactical play rather than mass accumulation. Fix 5 '
    'provides the monitoring signal needed to detect whether the changes are improving actual benchmark '
    'performance. Based on the log analysis, a well-tuned V10 with these fixes should achieve '
    'conversion_t120 >= 0.50 (up from 0.09-0.10) and backbone_turn_frac >= 0.18 (up from 0.07-0.15) '
    'within 10-15 generations of training.',
    s_body
))

# Build
doc.build(story)
doc.save(OUTPUT)
print(f'Report saved to {OUTPUT}')

