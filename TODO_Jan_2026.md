# Notes on next steps

## Dynamical systems and numerical analysis related stuff

- Discontinuties lurk in $d\theta/dt$ equation related to targets being in the 
blind spot and disappearing/reappearing. This normally is associated with 
unstable equilibria, but needs to be mathematically considered more carefully. 
Maybe a mollifier?

  *(2026-08 clarification: this is about the **genuine** blind spot — a weight
  whose support ends before the rear, e.g. `b_weight < π`, really does make
  targets wink out. Do not confuse it with the separate **wrapping-extent bug**
  fixed 2026-08-04, in which the closest target vanished from perception while
  straddling the rear branch cut even under full-support/uniform weight. That
  one was an interval-arithmetic error, not a modeling discontinuity; see
  CLAUDE.md "Common gotchas".)*
- [AJB] Related: the ODEs have some stiff behavior. One has only to look at the root 
finding situation for $d\gamma/dt$ to see an example of this. It's horribly 
unstable. The mathematics of this need to be explored and handled.

Note: to some degree, stiff behavior associated with targets located behind an 
observer is likely quite biological. I wouldn't call the equations being stiff 
there a bug - it just needs to not cause numerical problems.

- [JWB] Again related: the model is slow. We have to find the nearby stable equilibrium 
for $\gamma$ before we have a well defined ODE for $\theta$ that we can solve. 
Mathematically, this arrives from the separation of timescales assumption 
between neural activity and physical activity. Perhaps $\gamma$ doesn't have to 
actually be at equilibrium for this all to work, or perhaps there is a way to 
leverage the fact that you are probably close to an equilibrium already to 
speed things up. There's lots of nice analysis that can be done here, and if we 
are going to scale up to 100s of individuals, we are going to need to work out 
these wrinkles.

- From the point of view of dynamical systems analysis, the addition of signal 
strengths related to the precieved size of targets with positive area is wholly 
unexplored. And it's clearly different than the delta function target case. 
What are the geometries involved? What happens if you add a third target? Do you 
recover that infinite bifurcation chain which is broken when you change $\nu$ 
away from 1?

- Related to THAT: the published results assume that locusts turn infinitely 
fast to face the consensus direction. Now that we have a physical model of 
turning, the best-fit value for $\nu$ is probably different, and we may have an 
even better match to the experimental data.

## Modeling related thoughts

This framework gives us a model of attraction - great! But that is all. So it's 
likely that if we implement a ton of locusts with this where each other locust 
is a target and everyone has constant speed, they implode down into a frothy 
mass that goes nowhere. An external nudge may or may not give that some 
direction, but missing from this is a sense of repulsion.

But I would like to avoid repulsion. It's top-down, and really, the question is 
one of blocking (when does your target become an obstacle?). But I think there 
is an even lower-order thing we can do first. Here is my observation:
- When you are far away from a conspecific and feeling gregarious, that 
conspecific is attractive. Great, we have that.
- When you get close enough to that conspecific and they are moving somewhere, 
you *follow* them. That is, maintain a comfortable distance. Slow (not repel!) 
if you get too close, speed up if you get too far. We do this all the time when 
we are walking with our friends.

So, we have a torque model, I suggest all we need to mainatin some distance is 
a model that will adjust velocity based on object distance and angular location. 
That is, if you are close but next to me or behind me, I don't care (unless you 
are trying to eat me?). But if you are in front of me, I slow down, and once 
there is space again, I speed up. It doesn't provide a way to go around the 
blocking locust once you get fed up with them, but maybe we won't need that if 
the external nudge keeps the edge locusts moving somewhere...?

**Christopher calls dibs on this:** I am chomping at the bit to mathematically 
explore certain alterations to the Hamiltonian that might give us not only 
*attraction* but also *avoidance*. I have ulterior motives for this, related to 
larger things that fly.

## Python coding tasks

- I haven't tested the segment geometry in Targets in a long time. I place the 
chances it isn't broken in some way at less than 20%. Could be useful for locusts, 
which look more like segments than circles.

- Suppose we are simulating with, e.g., circular targets whose signal varies 
according to distance. Someone should look into being able to concurrently have 
targets whose signal does not vary, ideally delta function targets. This is 
because I would like to put targets on a circle of radius 1 billion or something 
similar as an external nudge. I think it would be interesting to be able to have 
more than one of them - particularly across from each other - so that locusts 
have a nudge toward an orientation but left/right in that orientation doesn't 
matter. The strength of these targets could be less than or equal to a locust 
that is $l_0$ away, so that it is always weaker than what is happening nearby.

- Scaling this up to a model of collective behavior is a big task. Here are my 
quick thoughts on it:
    - I think everyone can share a Targets object which gets updated with all 
    locust positions as long as each locust can then remove itself from the list 
    before conducting calculations.
    - Then we have a PerceptionModel and an IsingExtModel object. These in some 
    way encode a given locust's physical and mental states respectively, so they 
    probably have to be unique to each locust. From an ABM standpoint, this 
    probably isn't a big deal.
    - How do we do PiC with this? How do we make it not take forever?

If we can successfully transform this into a collective behavior model that 
doesn't take forever to run, we can throw it through our existing particle 
filter pipeline, which I call a paper: one that both describes our extensions to 
the existing Ising framework along with a data-driven comparison to alignment 
based models (as reported in our previously submitted paper).
