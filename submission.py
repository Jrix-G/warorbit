# ─────────────────────────────────────────────────────────────────────────────
# Orbit Wars — ProBot Submission
# ─────────────────────────────────────────────────────────────────────────────
import math, time
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

SUN_X, SUN_Y, SUN_RADIUS, INNER_ORBIT_R = 50.0, 50.0, 5.0, 30.0
MAX_TIME_MS = 900

def fleet_speed(ships): return min(1.0 + ships // 20, 6.0)

def fleet_hits_sun(sx, sy, angle, sr=SUN_RADIUS+1.5):
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = SUN_X-sx, SUN_Y-sy
    t = fx*dx + fy*dy
    if t < 0: return False
    return math.hypot(sx+t*dx-SUN_X, sy+t*dy-SUN_Y) < sr

class Planet:
    __slots__=("id","owner","x","y","radius","ships","production")
    def __init__(self,raw): self.id,self.owner,self.x,self.y,self.radius,self.ships,self.production=raw
    def dist(self,o): return math.hypot(self.x-o.x,self.y-o.y)
    def dist_xy(self,x,y): return math.hypot(self.x-x,self.y-y)
    def angle_to(self,o): return math.atan2(o.y-self.y,o.x-self.x)
    def angle_to_xy(self,x,y): return math.atan2(y-self.y,x-self.x)

class Fleet:
    __slots__=("id","owner","x","y","angle","from_planet_id","ships")
    def __init__(self,raw): self.id,self.owner,self.x,self.y,self.angle,self.from_planet_id,self.ships=raw

class GameState:
    def __init__(self,obs):
        g=lambda k: getattr(obs,k,None) or obs.get(k)
        self.my_id=g("player"); self.ang_vel=g("angular_velocity")
        self.step=g("step") or 0
        self.planets=[Planet(p) for p in g("planets")]
        self.fleets=[Fleet(f) for f in g("fleets")]
        self.comet_ids=set(g("comet_planet_ids") or [])
        self._pmap={p.id:p for p in self.planets}
        self.my_planets=[p for p in self.planets if p.owner==self.my_id]
        self.enemy_planets=[p for p in self.planets if p.owner not in(-1,self.my_id)]
        self.neutral_planets=[p for p in self.planets if p.owner==-1]
        self.incoming=defaultdict(lambda:defaultdict(int))
        for f in self.fleets:
            t=self._target(f)
            if t: self.incoming[t][f.owner]+=f.ships
    def _target(self,f):
        b,bd=None,9999
        for p in self.planets:
            a=math.atan2(p.y-f.y,p.x-f.x); diff=abs((a-f.angle+math.pi)%(2*math.pi)-math.pi)
            if diff<0.3:
                d=math.hypot(p.x-f.x,p.y-f.y)
                if d<bd: bd,b=d,p.id
        return b
    def planet(self,pid): return self._pmap.get(pid)
    def is_inner(self,p): return p.dist_xy(SUN_X,SUN_Y)<INNER_ORBIT_R
    def net_threat(self,p):
        inc=self.incoming.get(p.id,{})
        return sum(v for k,v in inc.items() if k!=self.my_id and k!=-1)-inc.get(self.my_id,0)
    def phase(self):
        r=len(self.my_planets)/max(len(self.planets),1)
        return "early" if r<0.2 else("late" if r>=0.55 else "mid")

class Predictor:
    def __init__(self,state): self.s=state
    def future_pos(self,p,t):
        if not self.s.is_inner(p): return p.x,p.y
        dx,dy=p.x-SUN_X,p.y-SUN_Y; r=math.hypot(dx,dy)
        a=math.atan2(dy,dx)+self.s.ang_vel*t
        return SUN_X+r*math.cos(a),SUN_Y+r*math.sin(a)
    def intercept(self,src,dst,ships):
        sp=fleet_speed(ships); tx,ty=dst.x,dst.y
        for _ in range(4):
            d=math.hypot(tx-src.x,ty-src.y); tx,ty=self.future_pos(dst,max(1,int(d/sp)))
        return tx,ty
    def aim(self,src,dst,ships):
        if not self.s.is_inner(dst): return src.angle_to(dst)
        tx,ty=self.intercept(src,dst,ships); return src.angle_to_xy(tx,ty)
    def eta(self,src,dst,ships):
        tx,ty=self.intercept(src,dst,ships)
        return max(1,int(math.hypot(tx-src.x,ty-src.y)/fleet_speed(ships)))
    def safe_aim(self,src,dst,ships):
        a=self.aim(src,dst,ships)
        if fleet_hits_sun(src.x,src.y,a):
            for d in[0.12,-0.12,0.25,-0.25,0.4,-0.4]:
                if not fleet_hits_sun(src.x,src.y,a+d): return a+d
        return a

class SP:
    __slots__=("id","owner","ships","production")
    def __init__(self,p): self.id,self.owner,self.ships,self.production=p.id,p.owner,p.ships,p.production
class SF:
    __slots__=("owner","tid","ships","eta")
    def __init__(self,o,t,s,e): self.owner,self.tid,self.ships,self.eta=o,t,s,e

class Simulator:
    def __init__(self,state,pred): self.s=state; self.p=pred
    def clone(self):
        P={p.id:SP(p) for p in self.s.planets}; F=[]
        for f in self.s.fleets:
            t=self.s._target(f)
            if t:
                tp=self.s.planet(t)
                if tp: F.append(SF(f.owner,t,f.ships,max(1,int(math.hypot(tp.x-f.x,tp.y-f.y)/fleet_speed(f.ships)))))
        return P,F
    @staticmethod
    def step(P,F):
        for p in P.values():
            if p.owner>=0: p.ships+=p.production
        R=[]
        for f in F:
            f.eta-=1
            if f.eta<=0:
                p=P[f.tid]
                if p.owner==f.owner: p.ships+=f.ships
                else:
                    p.ships-=f.ships
                    if p.ships<0: p.owner=f.owner; p.ships=abs(p.ships)
            else: R.append(f)
        F[:]=R
    def score(self,P,F,mi):
        ms=sum(p.ships for p in P.values() if p.owner==mi)+sum(f.ships for f in F if f.owner==mi)
        es=sum(p.ships for p in P.values() if p.owner not in(-1,mi))+sum(f.ships for f in F if f.owner not in(-1,mi))
        mp=sum(p.production for p in P.values() if p.owner==mi)
        ep=sum(p.production for p in P.values() if p.owner not in(-1,mi))
        return (ms-es)+40*(mp-ep)
    def eval(self,actions,steps=5):
        mi=self.s.my_id; P,F=self.clone()
        for fi,ti,sh in actions:
            src=P.get(fi); sp=self.s.planet(fi); dp=self.s.planet(ti)
            if src and sp and dp and src.ships>=sh>0:
                F.append(SF(mi,ti,sh,self.p.eta(sp,dp,sh))); src.ships-=sh
        for _ in range(steps): self.step(P,F)
        return self.score(P,F,mi)
    def best(self,cands,steps=5):
        bs,bc=-1e18,[]
        for c in cands:
            sc=self.eval(c,steps)
            if sc>bs: bs,bc=sc,c
        return bc

class Strategy:
    def __init__(self,state,pred): self.s=state; self.p=pred; self.mi=state.my_id
    def send(self,src,r=0.65):
        t=max(0,self.s.net_threat(src)); rv=max(src.production*3,t+6)
        return max(0,int((src.ships-rv)*r))
    def tscore(self,src,dst):
        d=src.dist(dst)
        if d<0.1: return -999
        val=dst.production*50-dst.ships+(20 if dst.owner==-1 else 60 if dst.owner!=self.mi else 0)
        return val/(d/fleet_speed(self.send(src) or 1)+1)
    def early(self):
        A=[]
        for src in sorted(self.s.my_planets,key=lambda p:-p.ships):
            b=self.send(src,0.75)
            for dst in sorted(self.s.neutral_planets,key=lambda t:self.tscore(src,t),reverse=True)[:4]:
                n=dst.ships+5
                if b>=n: A.append((src.id,dst.id,n)); b-=n
        return A
    def mid(self):
        A=[]
        for p in self.s.my_planets:
            if self.s.net_threat(p)>0:
                for d in sorted([x for x in self.s.my_planets if x.id!=p.id],key=lambda x:x.dist(p))[:2]:
                    s=min(self.s.net_threat(p)+6,self.send(d,0.5))
                    if s>0: A.append((d.id,p.id,s))
        if self.s.enemy_planets:
            tgt=min(self.s.enemy_planets,key=lambda p:p.ships); n=tgt.ships+10; rec=0
            for src in sorted(self.s.my_planets,key=lambda p:p.dist(tgt)):
                av=self.send(src,0.65)
                if av>0 and rec<n: s=min(av,n-rec); A.append((src.id,tgt.id,s)); rec+=s
        for src in self.s.my_planets:
            b=self.send(src,0.5)
            for dst in sorted(self.s.neutral_planets,key=lambda t:src.dist(t))[:2]:
                n=dst.ships+3
                if b>=n: A.append((src.id,dst.id,n)); b-=n
        return A
    def late(self):
        if not self.s.enemy_planets: return self.mid()
        A=[]; pri=max(self.s.enemy_planets,key=lambda p:p.production)
        for src in self.s.my_planets:
            s=self.send(src,0.85)
            if s>pri.production: A.append((src.id,pri.id,s))
        secs=[p for p in self.s.enemy_planets if p.id!=pri.id]
        if secs:
            for src in self.s.my_planets:
                sec=min(secs,key=lambda p:src.dist(p)); s=self.send(src,0.3)
                if s>sec.ships: A.append((src.id,sec.id,s))
        return A
    def aggro(self):
        if not self.s.enemy_planets: return []
        t=min(self.s.enemy_planets,key=lambda p:p.ships)
        return [(src.id,t.id,s) for src in self.s.my_planets for s in[self.send(src,0.9)] if s>0]
    def defend(self):
        if not self.s.my_planets: return []
        a=max(self.s.my_planets,key=lambda p:p.ships)
        return [(src.id,a.id,s) for src in self.s.my_planets if src.id!=a.id for s in[self.send(src,0.4)] if s>0]
    def candidates(self):
        ph={"early":self.early,"mid":self.mid,"late":self.late}[self.s.phase()]()
        return[ph,self.mid(),self.early(),self.aggro(),self.defend()]

def agent(obs,config=None):
    t0=time.time()
    try:
        state=GameState(obs)
        if not state.my_planets: return []
        pred=Predictor(state); sim=Simulator(state,pred); strat=Strategy(state,pred)
        best=sim.best(strat.candidates())
        moves=[]; spent={}
        for fi,ti,sh in best:
            src=state.planet(fi); dst=state.planet(ti)
            if not src or not dst or src.owner!=state.my_id: continue
            avail=src.ships-spent.get(fi,0)-1; send=min(sh,max(0,avail))
            if send<=0: continue
            angle=pred.safe_aim(src,dst,send)
            moves.append([fi,float(angle),int(send)]); spent[fi]=spent.get(fi,0)+send
            if(time.time()-t0)*1000>900: break
        return moves
    except: return []