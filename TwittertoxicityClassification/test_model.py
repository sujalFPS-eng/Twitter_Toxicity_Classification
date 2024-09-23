from Toxicity import Model

model = Model()

print(model.score("This sentence is not toxic. Like not toxic at all!"))

sentences = []
sentences.append('Hello, world!')
sentences.append('I hate this world! And I am so toxic and angry!')
sentences.append("Ahhhhhhhhh")

predictions = model.score(sentences)

for i in range(len(sentences)):
    print(f'{predictions[i]}: {sentences[i]}')