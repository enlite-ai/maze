# In a core env (which maintains a pubsub broker)

# Create a topic
inventory_events = self.pubsub.create_event_topic(InventoryEvents)

# Register a subscriber (can be a reward aggregator
# or any other class implementing the Subscriber interface)
self.pubsub.register_subscriber(my_subscriber)

# Dispatch an event
inventory_events.piece_discarded(piece=(50, 10))
