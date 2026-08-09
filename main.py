from scripts.classifier import MCCClassifier
import json


def main():

    classifier = MCCClassifier()

    print("=" * 60)
    print("Wikipedia Entity Understanding")
    print("=" * 60)

    while True:

        page_name = input("\nEnter Wikipedia article title (or type 'exit'): ").strip()

        if page_name.lower() == "exit":
            print("\nGoodbye!")
            break

        result = classifier.classify(page_name)

        print("\nKnowledge Retrieved")
        print("-" * 40)
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
