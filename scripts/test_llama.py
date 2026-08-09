from scripts.classifier import MCCClassifier

classifier = MCCClassifier()

result = classifier.classify(
    merchant_name="Starbucks",
    description="Coffeehouse serving coffee, tea and snacks."
)

print(result)