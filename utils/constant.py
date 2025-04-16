NODE_TYPES = ["Product", "Ingredient"]

RELATIONSHIP_TYPES = [
    ("Product", "HAS", "Ingredient"),
    ("Product", "HARMFUL", "Ingredient"),
    ("Product", "POSITIVE", "Ingredient"),
    ("Product", "NOTABLE", "Ingredient"),
]
