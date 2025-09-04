# ------------------------------------------------------------------------------------------------
# event_bus.py - The Nervous System of the Application
# ------------------------------------------------------------------------------------------------
#
# This is a simple implementation of the publish-subscribe pattern.
# It allows components to communicate with each other without being directly
# aware of each other (decoupling).
#
# - A component can "subscribe" a function to an event type.
# - Another component can "publish" an event, and the bus calls all
#   subscribed functions.
#
# ------------------------------------------------------------------------------------------------

class EventBus:
    def __init__(self):
        self.listeners = {}

    def subscribe(self, event_type, listener):
        """
        Subscribe a listener function to an event type.
        Args:
            event_type (str): The name of the event (e.g., 'TICK', 'CANDLE_CLOSED_15MIN').
            listener (callable): The function to be called when the event is published.
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)

    def publish(self, event_type, *args, **kwargs):
        """
        Publish an event to all subscribed listeners.
        Args:
            event_type (str): The name of the event to publish.
            *args, **kwargs: The arguments to pass to the listener functions.
        """
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                try:
                    listener(*args, **kwargs)
                except Exception as e:
                    # To prevent one bad listener from crashing the whole system
                    print(f"Error in listener for event '{event_type}': {e}")
