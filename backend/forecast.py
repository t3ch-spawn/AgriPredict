from xgb_training_initial_2 import predict_future_price

examples = [
    ('Rice (local)', 'Borno', 2026, 3),
    ('Rice (local)', 'Yobe', 2026, 4),
    # ('Rice (local)', 'Yobe', 2026, 5),
    # ('Rice (local)', 'Yobe', 2026, 6),
    # ('Yam', 'Borno', 2026, 6),
    # ('Beans (red)', 'Yobe', 2026, 9),
    # ('Oranges', 'Adamawa', 2027, 1),
    # ('Tomatoes', 'Borno', 2027, 11)
]

for commodity, state, year, month in examples:
    result = predict_future_price(commodity, state, year, month)
    print(f"{result['commodity']} in {result['state']} ({year}-{result['month']:02d}):")
    print(f"  Predicted Price: ₦{result['predicted_price']}")
    print(f"  Recent Price: ₦{result['recent_price']}")
    print(f"  Model Type: {result['model_type']}")
    print(f"  Confidence: {result['confidence']}\n")