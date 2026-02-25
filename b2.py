from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Structure: Domain -> Flow <- Intent
model = DiscreteBayesianNetwork([
    ('Domain', 'Flow'),
    ('Intent', 'Flow')
])

# Domain prior
cpd_domain = TabularCPD(
    variable='Domain', variable_card=2,
    values=[[0.8], [0.2]]   # URS, Other
)

# Intent prior
cpd_intent = TabularCPD(
    variable='Intent', variable_card=2,
    values=[[0.6], [0.4]]   # generate_urs, other
)

# Flow | Domain, Intent
cpd_flow = TabularCPD(
    variable='Flow', variable_card=2,
    values=[
        [0.9, 0.3, 0.4, 0.05],  # Flow = Yes
        [0.1, 0.7, 0.6, 0.95]   # Flow = No
    ],
    evidence=['Domain', 'Intent'],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_domain, cpd_intent, cpd_flow)
print(model.check_model())