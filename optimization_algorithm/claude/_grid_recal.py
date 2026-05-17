"""Grid-search recalibration using segment-based constraint values + cfg-based scoring."""
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

def sim_one(idx):
    row = df.iloc[idx]; ort = _sf(row.get("ORT")) or 850
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
    return ec.extract_from_state(state,sim.basic_info,rt), row['TS'], row['EXT']

print("Caching sim_T for 30 records...")
data={}
for idx in BEST+MED+WORST:
    print(" idx=%d..."%idx,end=' ',flush=True)
    data[idx]=sim_one(idx)
print("done\n")

def score_all(cfg_override=None):
    if cfg_override:
        old={k:ec.CONSTRAINT_CFG.get(k) for k in cfg_override}
        for k,v in cfg_override.items(): ec.CONSTRAINT_CFG[k]=v
    tiers={'B':[],'M':[],'W':[]}
    for idx in BEST:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['B'].append(p-ec.W_BONUS*b)
    for idx in MED:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['M'].append(p-ec.W_BONUS*b)
    for idx in WORST:
        v,ts,ext=data[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['W'].append(p-ec.W_BONUS*b)
    if cfg_override:
        for k,v in old.items():
            if v is not None: ec.CONSTRAINT_CFG[k]=v
    return tiers

# Show initial
t0=score_all()
print("Initial: BEST %.0f-%.0f MED %.0f-%.0f WORST %.0f-%.0f"%(
    min(t0['B']),max(t0['B']),min(t0['M']),max(t0['M']),min(t0['W']),max(t0['W'])))

# Fine grid around best values from previous run
best_cfg=None; best_ok=0
for m_s7 in [0.004,0.005,0.006,0.007,0.008,0.009]:
 for m_s8 in [0.3,0.4,0.5,0.6,0.7,0.8]:
  for m_s9 in [0.005,0.008,0.01,0.012,0.015]:
   for m_h3 in [0.005,0.008,0.01,0.012,0.015]:
    for m_s13 in [0.05,0.08,0.1,0.12,0.15]:
     cfg={'M_S7':m_s7,'M_S8':m_s8,'M_S9':m_s9,'M_H3':m_h3,'M_S13':m_s13}
     t=score_all(cfg)
     bmax,mmax,mmin,wmin,wmax=max(t['B']),max(t['M']),min(t['M']),min(t['W']),max(t['W'])
     ok=0
     if bmax<=9: ok+=1
     if 10<=mmin and mmax<=29: ok+=1
     if 30<=wmin and wmax<=40: ok+=1
     if ok>=best_ok:
         spread=max(t['B'])-min(t['B'])+max(t['M'])-min(t['M'])+max(t['W'])-min(t['W'])
         if ok>best_ok or (best_cfg is None or spread<best_spread):
             best_ok=ok; best_cfg=cfg; best_spread=spread
             best_tiers=t

print("\nBest: %d/3 tiers OK"%best_ok)
if best_ok>0:
    print("BEST: %s" % ['%.1f'%s for s in best_tiers['B']])
    print("MED:  %s" % ['%.1f'%s for s in best_tiers['M']])
    print("WORST:%s" % ['%.1f'%s for s in best_tiers['W']])
    print("Config:",best_cfg)
    # Apply
    for k,v in best_cfg.items(): ec.CONSTRAINT_CFG[k]=v
    print("Applied to CONSTRAINT_CFG")
    # Save
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'_grid_result.txt'),'w') as f:
        f.write("Grid recalibration result:\n%s\n" % str(best_cfg))
        f.write("BEST: %s\n"%best_tiers['B'])
        f.write("MED: %s\n"%best_tiers['M'])
        f.write("WORST: %s\n"%best_tiers['W'])
