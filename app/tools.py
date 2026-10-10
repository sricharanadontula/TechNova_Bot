from langchain_core.tools import tool


ORDERS = {
    "TN1001": {
        "status": "Processing",
        "expected_delivery": "October 10, 2026",
        "item": "Wireless Headphones"
    },
    "TN1002": {
        "status": "Shipped",
        "expected_delivery": "October 8, 2026",
        "item": "Smart Watch"
    },
    "TN1003": {
        "status": "Delivered",
        "expected_delivery": "October 3, 2026",
        "item": "Bluetooth Speaker"
    }
}


POLICIES = {
    "returns": "TechNova accepts returns within 30 days of delivery. The product must be unused and in its original packaging.",
    
    "warranty": "TechNova products include a one-year limited warranty covering manufacturing defects.",
    
    "shipping": "Standard shipping usually takes 3 to 5 business days. Express shipping usually takes 1 to 2 business days."
}

@tool
def get_order_status(order_id: str) -> str:
    """Look up the status, item, and expected delivery date for a TechNova order."""
    
    order = ORDERS.get(order_id)

    if order is None:
        return f"No order was found with order ID {order_id}."

    return (
        f"Order {order_id}: "
        f"Item: {order['item']}, "
        f"Status: {order['status']}, "
        f"Expected delivery: {order['expected_delivery']}."
    )

@tool
def get_policy(topic: str) -> str:
    """Look up TechNova policies for returns, warranty, or shipping."""
    
    topic = topic.lower().strip()

    policy = POLICIES.get(topic)

    if policy is None:
        return (
            f"No policy was found for '{topic}'. "
            "Available topics are returns, warranty, and shipping."
        )

    return policy

if __name__ == "__main__":
    print(get_order_status.invoke({"order_id": "TN1002"}))
    print(get_policy.invoke({"topic": "returns"}))