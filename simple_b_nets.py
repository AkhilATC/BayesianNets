from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

"""
Rain → Traffic
"""

model = BayesianNetwork(["Rain","Traffic"])

# Prior probability of Rain
cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[[0.7], [0.3]]  # No rain, Rain
)

# Traffic depends on Rain
cpd_traffic = TabularCPD(
    variable='Traffic',
    variable_card=2,
    values=[[0.9, 0.2],   # No traffic
            [0.1, 0.8]],  # Traffic
    evidence=['Rain'],
    evidence_card=[2]
)

model.add_cpds(cpd_rain, cpd_traffic)
model.check_model()

# Inference
infer = VariableElimination(model)
print(infer.query(['Traffic'], evidence={'Rain':1}))


# Note

cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[[0.7], [0.3]]  # No rain, Rain
)
"""
variable_card = 2   
cardinality = number of possible values
Rain can be:
    1. (0 = No Rain)

    2. (1 = Rain)
So cardinality = 2.
"""
cpd_traffic = TabularCPD(
    variable='Traffic',
    variable_card=2,
    values=[[0.9, 0.2],   # No traffic
            [0.1, 0.8]],  # Traffic
    evidence=['Rain'],
    evidence_card=[2]
)

"""
values = [
    [0.9, 0.2],  # Traffic = 0 (No traffic)
    [0.1, 0.8]   # Traffic = 1 (Traffic)
]
            Rain
          0      1
Traffic 0 0.9    0.2
Traffic 1 0.1    0.8

What is the probability of Rain given Traffic?
P(Rain∣Traffic)
prior :
P(Rain=0)=0.7,P(Rain=1)=0.3


from pgmpy.inference import VariableElimination

infer = VariableElimination(model)
posterior = infer.query(['Rain'], evidence={'Traffic':1})
print(posterior)
"""