"""
Order Validation

Validates trading signals according to the StrategySignal schema.
"""

from typing import Dict, Any, Tuple, Optional
import re


class ValidationError(Exception):
    """Raised when a signal fails validation."""
    pass


class SignalValidator:
    """Validates trading signals against the StrategySignal schema."""
    
    # Valid symbols that the orchestrator can trade
    VALID_SYMBOLS = {"AAPL", "NVDA", "SPY", "QQQ", "BTC", "ETH"}
    
    # Valid action types
    VALID_ACTIONS = {"buy", "sell"}
    
    # Valid order types
    VALID_ORDER_TYPES = {"market", "limit"}
    
    # Valid time-in-force values
    VALID_TIME_IN_FORCE = {"day", "gtc", "ioc", "fok"}
    
    @classmethod
    def validate(cls, signal: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validate a signal dictionary.
        
        Args:
            signal: Signal dict to validate, or None
            
        Returns:
            (is_valid, error_message) tuple
        """
        if signal is None:
            # None is valid - it means skip this minute
            return True, None
        
        if not isinstance(signal, dict):
            return False, "Signal must be a dictionary"
        
        # Required fields
        required_fields = {"symbol", "action", "quantity", "price"}
        missing = required_fields - set(signal.keys())
        if missing:
            return False, f"Missing required fields: {missing}"
        
        # Validate symbol
        symbol = signal.get("symbol")
        if not isinstance(symbol, str):
            return False, "symbol must be a string"
        if symbol not in cls.VALID_SYMBOLS:
            return False, f"symbol '{symbol}' not in valid symbols: {cls.VALID_SYMBOLS}"
        
        # Validate action
        action = signal.get("action")
        if not isinstance(action, str) or action not in cls.VALID_ACTIONS:
            return False, f"action must be 'buy' or 'sell', got '{action}'"
        
        # Validate quantity
        try:
            quantity = float(signal.get("quantity"))
            if quantity <= 0:
                return False, f"quantity must be positive, got {quantity}"
        except (TypeError, ValueError):
            return False, f"quantity must be a positive number, got {signal.get('quantity')}"
        
        # Validate price
        try:
            price = float(signal.get("price"))
            if price <= 0:
                return False, f"price must be positive, got {price}"
        except (TypeError, ValueError):
            return False, f"price must be a positive number, got {signal.get('price')}"
        
        # Optional fields with defaults
        order_type = signal.get("order_type", "market")
        if order_type not in cls.VALID_ORDER_TYPES:
            return False, f"order_type must be 'market' or 'limit', got '{order_type}'"
        
        time_in_force = signal.get("time_in_force", "day")
        if time_in_force not in cls.VALID_TIME_IN_FORCE:
            return False, f"time_in_force must be one of {cls.VALID_TIME_IN_FORCE}, got '{time_in_force}'"
        
        # Optional confidence field
        if "confidence" in signal:
            try:
                confidence = float(signal.get("confidence"))
                if not 0.0 <= confidence <= 1.0:
                    return False, f"confidence must be between 0.0 and 1.0, got {confidence}"
            except (TypeError, ValueError):
                return False, f"confidence must be a number between 0.0 and 1.0"
        
        # Optional reason field
        if "reason" in signal:
            reason = signal.get("reason")
            if not isinstance(reason, str):
                return False, "reason must be a string"
        
        return True, None
    
    @classmethod
    def validate_strict(cls, signal: Optional[Dict[str, Any]]) -> None:
        """
        Validate a signal and raise ValidationError if invalid.
        
        Args:
            signal: Signal to validate
            
        Raises:
            ValidationError: If signal is invalid
        """
        is_valid, error_msg = cls.validate(signal)
        if not is_valid:
            raise ValidationError(error_msg)
