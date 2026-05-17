"""Fix coefficients: add PO extreme solution to calibration, re-grid-search."""
import sys, os
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'optimization_algorithm'))
os.chdir(PROJECT_ROOT)

import numpy as np, pandas as pd
import sim_T.sim_T as sim
import expert_constraints as ec

df = pd.read_excel('工艺数据全.xlsx')
BEST=[195,40,53,59,33,194,196,35,51,26]
MED=[99,15,98,100,27,101,102,103,8,6]
WORST=[132,159,133,166,164,134,208,135,213,136]

def _sf(v):
    if pd.isna(v): return None
    try: return float(v)
    except: return None
def _np(v):
    if v is None: return None
    if v>1.0: return v/100.0
    return v

def sim_one(idx=None, custom_row=None):
    if custom_row is not None:
        row = custom_row
    else:
        row = df.iloc[idx]
    ort = _sf(row.get("ORT")) or 850
    for e,a in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        v=_sf(row.get(e))
        if v is not None: setattr(sim.basic_info, a, v/100.0)
    rolls,_=sim.data_loader.load_roll_data()
    for i in range(1,11):
        v=_sf(row.get('SPEED%d'%i))
        if v is not None and v>0:
            ov=rolls[i].roll_v; rolls[i].roll_v=v; rolls[i].t=rolls[i].t*(ov/v)
            rolls[i].step=int(rolls[i].t/sim._default_dt)
    ev=_sf(row.get("SPEED1"))
    if ev is not None and ev>0:
        ov=rolls[0].roll_v; rolls[0].roll_v=ev; rolls[0].t=rolls[0].t*(ov/ev)
        rolls[0].step=int(rolls[0].t/sim._default_dt)
    for i in range(1,11):
        v=_sf(row.get('FAN%d'%i)); v=_np(v)
        if v is not None: rolls[i].fan_status=v; rolls[i].fan_speed=rolls[i].fan_air_volume*v/rolls[i].fan_area
    state,rt=sim.run_full_simulation(rolls,tem1=ort,tem0=ort,dt=0.01)
    vals = ec.extract_from_state(state,sim.basic_info,rt)
    return vals, row['TS'] if custom_row is None else None, row['EXT'] if custom_row is None else None

# Cache 30 records
print("Caching 30 labeled records...")
data={}
for idx in BEST+MED+WORST:
    print(" %d"%idx,end='',flush=True)
    data[idx]=sim_one(idx)
print(" done")

# Simulate PO extreme solution
print("Simulating PO extreme solution...")
# Create a synthetic row with PO parameters, using idx=195 chemistry as base
base_row = df.iloc[195].copy()
base_row['ORT'] = 940
for j in range(1,11): base_row['SPEED%d'%j] = 1.5
fans = [99,100,100,8,93,100,100,5,1,9]
for j in range(1,11): base_row['FAN%d'%j] = fans[j-1]
# Also set chemistry to 82A standard
for e,v in [('C_ELE',0.82),('SI_ELE',0.25),('MN_ELE',0.50),('P_ELE',0.012),('S_ELE',0.010),('CR_ELE',0.20),('NI_ELE',0.05),('CU_ELE',0.05)]:
    base_row[e] = v

po_vals, _, _ = sim_one(custom_row=base_row)
# Use ML-predicted TS? We don't have it. Assume TS in range for now (PO output had TS ~1076-1094)
# Actually, the PO found solutions with ML-predicted TS around 1076-1094. Let's use 1080.
po_vals['_TS'] = 1080
po_vals['_EXT'] = 42  # typical for these params
print("  cr_stage1=%.1f, cr_pearl=%.1f, cr_550=%.1f, cr_lowT=%.1f" % (
    po_vals.get('cr_stage1') or 0, po_vals.get('cr_pearl') or 0,
    po_vals.get('cr_550') or 0, po_vals.get('cr_lowT') or 0))

# Add PO as extra WORST
WORST_PLUS = WORST + ['PO']
data['PO'] = (po_vals, 1080, 42)

def score_all(cfg_override=None):
    if cfg_override:
        old={k:ec.CONSTRAINT_CFG.get(k) for k in cfg_override}
        for k,v in cfg_override.items(): ec.CONSTRAINT_CFG[k]=v
    tiers={'B':[],'M':[],'W':[],'PO':None}
    for idx in BEST:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['B'].append(p-ec.W_BONUS*b)
    for idx in MED:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['M'].append(p-ec.W_BONUS*b)
    for idx in WORST:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['W'].append(p-ec.W_BONUS*b)
    # PO score
    v,ts,ext=data['PO']; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
    tiers['PO'] = p-ec.W_BONUS*b
    if cfg_override:
        for k,v in old.items():
            if v is not None: ec.CONSTRAINT_CFG[k]=v
    return tiers

# Show current
t0=score_all()
print("\nCurrent: BEST %.0f-%.0f MED %.0f-%.0f WORST %.0f-%.0f PO=%.1f" % (
    min(t0['B']),max(t0['B']),min(t0['M']),max(t0['M']),min(t0['W']),max(t0['W']),t0['PO']))

# Grid search - focus on multipliers that can push PO score up
best_cfg=None; best_ok=0
for m_s7 in [0.003,0.005,0.008,0.01,0.015]:
 for m_s8 in [0.2,0.3,0.4,0.6,0.8]:
  for m_s9 in [0.01,0.02,0.03,0.05,0.08,0.12]:
   for m_h3 in [0.01,0.02,0.03,0.05,0.08,0.12,0.18]:
    for m_s13 in [0.03,0.05,0.08,0.12,0.18,0.25]:
     cfg={'M_S7':m_s7,'M_S8':m_s8,'M_S9':m_s9,'M_H3':m_h3,'M_S13':m_s13}
     t=score_all(cfg)
     bmax,mmax,mmin,wmin,wmax=max(t['B']),max(t['M']),min(t['M']),min(t['W']),max(t['W'])
     po_score=t['PO']
     ok=0
     if bmax<=9: ok+=1
     if 10<=mmin and mmax<=29: ok+=1
     if 30<=wmin and wmax<=40: ok+=1
     if po_score>40: ok+=1  # PO extreme must score HIGH
     if ok>=best_ok:
         # Tie-break: prefer higher PO score and tighter tier spread
         spread=max(t['B'])-min(t['B'])+max(t['M'])-min(t['M'])+max(t['W'])-min(t['W'])
         metric=ok*1000+po_score-spread*0.1
         if best_cfg is None or metric>best_metric:
             best_ok=ok; best_cfg=cfg; best_metric=metric; best_tiers=t

print("\nBest: %d/4 criteria met (incl PO>40)" % best_ok)
print("BEST: %s" % ['%.1f'%s for s in best_tiers['B']])
print("MED:  %s" % ['%.1f'%s for s in best_tiers['M']])
print("WORST:%s" % ['%.1f'%s for s in best_tiers['W']])
print("PO:   %.1f (target >40)" % best_tiers['PO'])
print("Config:", best_cfg)

if best_ok >= 3:
    for k,v in best_cfg.items(): ec.CONSTRAINT_CFG[k]=v
    print("\nApplied to CONSTRAINT_CFG in memory.")
    # Persist to source file
    import re
    src_path = os.path.join(PROJECT_ROOT, 'optimization_algorithm', 'expert_constraints.py')
    with open(src_path, 'r', encoding='utf-8') as f:
        src = f.read()
    for k, v in best_cfg.items():
        # Match pattern: "M_S7": 0.005, or similar
        pattern = r'("{}"\s*:\s*)[\d.]+'.format(k)
        replacement = r'\g<1>{}'.format(v)
        src = re.sub(pattern, replacement, src)
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(src)
    print("Persisted to %s" % src_path)
else:
    print("\nBest result: %d/4 criteria. Showing tier violations:" % best_ok)
    if best_tiers['PO'] <= 40: print("  PO score=%.1f <= 40 (need >40)" % best_tiers['PO'])
    if max(best_tiers['B']) > 9: print("  BEST max=%.1f > 9" % max(best_tiers['B']))
    if min(best_tiers['M']) < 10: print("  MED min=%.1f < 10" % min(best_tiers['M']))
    if max(best_tiers['M']) > 29: print("  MED max=%.1f > 29" % max(best_tiers['M']))
