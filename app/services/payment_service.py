"""Payment provider abstraction (manual mode active, Stripe pluggable later)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PaymentIntent:
    provider: str
    reference: str
    status: str


class PaymentProvider:
    def create_intent(self, plan_code: str, transaction_ref: Optional[str] = None) -> PaymentIntent:
        raise NotImplementedError


class ManualPaymentProvider(PaymentProvider):
    def create_intent(self, plan_code: str, transaction_ref: Optional[str] = None) -> PaymentIntent:
        return PaymentIntent(
            provider="manual",
            reference=transaction_ref or f"manual:{plan_code}",
            status="pending",
        )


def get_payment_provider() -> PaymentProvider:
    return ManualPaymentProvider()
