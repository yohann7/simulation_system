"""Auto-calibrate constraint multipliers. Iterates: PO -> analyze deviations -> adjust -> repeat.
Rule: NO new constraints. Only adjust existing multipliers in CONSTRAINT_CFG.
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'optimization_algorithm'))
os.chdir(PROJECT_ROOT)

# Import order matters: sim_T.sim_T must be first
import sim_T.sim_T as sim
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'sim_T'))
import calculate_all_sim_T as calc
_orig = calc.run_all_simulations
def _single(*a,**kw): kw['n_workers']=1; return _orig(*a,**kw)
calc.run_all_simulations = _single

import numpy as np, pandas as pd
import expert_constraints as ec
import PO

# ============================================================
# Labeled data (one-time sim_T cache)
# ============================================================
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

def sim_labeled(idx):
    row=df.iloc[idx]; ort=_sf(row.get("ORT")) or 850
    for e,a in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        v=_sf(row.get(e))
        if v is not None: setattr(sim.basic_info,a,v/100.0)
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

print("Caching 30 labeled records...")
cache={}
for idx in BEST+MED+WORST:
    print(" %d"%idx,end='',flush=True)
    cache[idx]=sim_labeled(idx)
print(" done\n")

def compute_tier_scores():
    tiers={'B':[],'M':[],'W':[]}
    for idx in BEST:
        v,ts,ext=cache[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['B'].append(p-ec.W_BONUS*b)
    for idx in MED:
        v,ts,ext=cache[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['M'].append(p-ec.W_BONUS*b)
    for idx in WORST:
        v,ts,ext=cache[idx]; _,p,b,_=ec.evaluate_constraints(v,pred_TS=ts,pred_Z=ext)
        tiers['W'].append(p-ec.W_BONUS*b)
    return tiers

# BEST reference statistics (for comparison)
BEST_REF={}
for j in range(1,11):
    vals=[df.iloc[i]['SPEED%d'%j] for i in BEST]
    BEST_REF['SPEED%d'%j]={'mean':float(np.mean(vals)),'std':float(np.std(vals)),'min':float(np.min(vals)),'max':float(np.max(vals))}
BEST_REF['ORT']={'mean':float(np.mean([df.iloc[i]['ORT'] for i in BEST])),'std':float(np.std([df.iloc[i]['ORT'] for i in BEST]))}

# Initial tier scores
t0=compute_tier_scores()
print("Initial tiers: BEST %.1f-%.1f, MED %.1f-%.1f, WORST %.1f-%.1f" % (
    min(t0['B']),max(t0['B']),min(t0['M']),max(t0['M']),min(t0['W']),max(t0['W'])))

# ============================================================
# Iterative calibration
# ============================================================
cfg=ec.CONSTRAINT_CFG
MAX_ROUNDS=6
po_max_iter=20

for round_num in range(1,MAX_ROUNDS+1):
    print("\n"+"="*60)
    print("ROUND %d (PO MaxIter=%d)"%(round_num,po_max_iter))

    # Run PO
    lb,ub,dim=PO.get_search_bounds()
    np.random.seed(42+round_num)
    try:
        best_X,best_cost,conv=PO.puma_optimize(
            nSol=12,MaxIter=po_max_iter,lb=lb,ub=ub,dim=dim,
            CostFunction=PO.stelmor_sim_CostFunction,
            BatchCostFunction=PO.stelmor_batch_cost_function,patience=10)
    except Exception as e:
        print("PO FAILED: %s"%e); break

    print("PO best cost: %.2f" % best_cost)

    # Extract PO params
    po_ort=best_X[0]; po_spd=[best_X[j] for j in range(1,11)]; po_fan=[best_X[10+j] for j in range(1,11)]

    # Check tier scores
    tiers=compute_tier_scores()
    bmax,mmax,mmin,wmin,wmax=max(tiers['B']),max(tiers['M']),min(tiers['M']),min(tiers['W']),max(tiers['W'])
    print("Tiers: BEST %.1f-%.1f, MED %.1f-%.1f, WORST %.1f-%.1f" % (
        min(tiers['B']),bmax,mmin,mmax,wmin,wmax))

    # Analyze deviations
    adjustments=[]

    # 1. ORT too high?
    ort_dev=po_ort-BEST_REF['ORT']['mean']
    if ort_dev>30:
        # ORT near upper bound - increase H3/S13 to penalize high-ORT strategies
        # (high ORT gives more heat, compensates for low fans)
        cfg['M_H3']=min(cfg['M_H3']*1.15,1.0)
        cfg['M_S13']=min(cfg['M_S13']*1.15,1.0)
        adjustments.append("ORT=%.0f too high (dev=%+.0f) -> M_H3*1.15, M_S13*1.15"%(po_ort,ort_dev))

    # 2. SPEEDs too high? (all > 1.4 = severe overspeed)
    overspeed_count=sum(1 for s in po_spd if s>1.4)
    if overspeed_count>=6:
        # Reduce B5 bonus (makes overspeed less rewarded) + increase S10 fast-side
        cfg['B_B5']=max(cfg['B_B5']*0.85,0.2)
        cfg['M_S10']=min(cfg['M_S10']*1.2,1.0)
        adjustments.append("%d speeds>1.4 -> B_B5*0.85, M_S10*1.2"%overspeed_count)

    # 3. Cost too negative? (bonuses dominating)
    if best_cost<-2:
        ec.W_BONUS=max(ec.W_BONUS*0.9,0.15)
        adjustments.append("Cost=%.1f too negative -> W_BONUS*0.9=%.2f"%(best_cost,ec.W_BONUS))

    # 4. Tier range check
    if bmax>9:
        cfg['M_S9']=min(cfg['M_S9']*1.15,5.0)
        adjustments.append("BEST max=%.1f>9 -> M_S9*1.15"%bmax)
    if mmin<10:
        cfg['M_S7']=min(cfg['M_S7']*1.2,0.1)
        adjustments.append("MED min=%.1f<10 -> M_S7*1.2"%mmin)
    if mmax>29:
        cfg['M_S7']=max(cfg['M_S7']*0.9,0.001)
        adjustments.append("MED max=%.1f>29 -> M_S7*0.9"%mmax)
    if wmin<30:
        cfg['M_S8']=min(cfg['M_S8']*1.15,5.0)
        cfg['M_S9']=min(cfg['M_S9']*1.1,5.0)
        adjustments.append("WORST min=%.1f<30 -> M_S8*1.15, M_S9*1.1"%wmin)
    if wmax>40:
        cfg['M_S8']=max(cfg['M_S8']*0.9,0.01)
        adjustments.append("WORST max=%.1f>40 -> M_S8*0.9"%wmax)
    if bmax>=mmin:
        cfg['M_S7']=min(cfg['M_S7']*1.15,0.1)
        cfg['M_S9']=min(cfg['M_S9']*1.15,5.0)
        adjustments.append("BEST-MED overlap -> M_S7*1.15, M_S9*1.15")

    # Clamp
    for k in ['M_H2','M_H3','M_S1','M_S2','M_S4','M_S7','M_S8','M_S9','M_S10','M_S11','M_S13']:
        if k in cfg: cfg[k]=max(0.001,min(cfg[k],100.0))
    ec.W_BONUS=max(0.1,min(ec.W_BONUS,0.5))
    cfg['B_B5']=max(0.1,min(cfg['B_B5'],10.0))

    if adjustments:
        print("Adjustments:")
        for a in adjustments: print("  "+a)
    else:
        print("No adjustments needed.")

    # Convergence check
    speed_ok=all(abs(po_spd[j]-BEST_REF['SPEED%d'%(j+1)]['mean'])<BEST_REF['SPEED%d'%(j+1)]['std']*2 for j in range(10))
    ort_ok=abs(po_ort-BEST_REF['ORT']['mean'])<BEST_REF['ORT']['std']*2
    tiers_ok=(bmax<=9 and mmin>=10 and mmax<=29 and wmin>=30 and wmax<=40 and bmax<mmin and mmax<wmin)

    if speed_ok and ort_ok and tiers_ok:
        print("CONVERGED!")
        break

    # Update MaxIter
    if round_num>=2 and best_cost>=prev_cost-0.01:
        po_max_iter=min(po_max_iter+15,80)
    prev_cost=best_cost

# Final persist
import re
src_path=os.path.join(PROJECT_ROOT,'optimization_algorithm','expert_constraints.py')
with open(src_path,'r',encoding='utf-8') as f: src=f.read()
for k in ['M_H3','M_S13','M_S7','M_S8','M_S9','M_S10','B_B5']:
    if k in cfg:
        src=re.sub(r'("{}"\s*:\s*)[\d.]+'.format(k),r'\g<1>{}'.format(cfg[k]),src)
with open(src_path,'w',encoding='utf-8') as f: f.write(src)
# W_BONUS (module-level, not in CONSTRAINT_CFG)
src2=open(src_path,'r',encoding='utf-8').read()
src2=re.sub(r'W_BONUS\s*=\s*[\d.]+','W_BONUS = %.3f'%ec.W_BONUS,src2)
with open(src_path,'w',encoding='utf-8') as f: f.write(src2)

print("\nFinal config persisted.")
print("Tiers: BEST %.1f-%.1f, MED %.1f-%.1f, WORST %.1f-%.1f" % (
    min(tiers['B']),max(tiers['B']),min(tiers['M']),max(tiers['M']),min(tiers['W']),max(tiers['W'])))
