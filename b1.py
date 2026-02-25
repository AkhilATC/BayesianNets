from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define graph
model = DiscreteBayesianNetwork([
    ('Domain', 'Intent'),
    ('Domain', 'Action'),
    ('Intent', 'Action')
])

# Domain CPD
cpd_domain = TabularCPD(
    variable='Domain', variable_card=2,
    values=[[0.8],   # URS
            [0.2]]   # Other
)

# Intent | Domain
cpd_intent = TabularCPD(
    variable='Intent', variable_card=2,
    values=[
        [0.6, 0.2],  # generate_urs
        [0.4, 0.8]   # other
    ],
    evidence=['Domain'], evidence_card=[2]
)

# Action | Domain, Intent
cpd_action = TabularCPD(
    variable='Action', variable_card=2,
    values=[
        # Flow probabilities
        [0.9, 0.3, 0.4, 0.05],
        # NoFlow
        [0.1, 0.7, 0.6, 0.95]
    ],
    evidence=['Domain', 'Intent'],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_domain, cpd_intent, cpd_action)
print(model.check_model())