from core import SimpleLinearModel

model = SimpleLinearModel()

X = [1,2,3]
y = [2,6,12]

model.train(X, y)
print(model.predict(2))  # Should print close to 12
print(model.predict(3))  # Should print close to 12
print(model.predict(4))  # Should print close to 12

model.save("model.pkl")

loaded = SimpleLinearModel.load("model.pkl")
print(loaded.predict(7))  # Should print close to 14
