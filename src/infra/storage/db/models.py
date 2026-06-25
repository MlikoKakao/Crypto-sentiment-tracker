from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    String,
    Integer,
    DateTime,
    Float,
    Text,
    null,
)


class Base(DeclarativeBase):
    pass


class Price(Base):
    __tablename__ = "prices"

    coin: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    price: Mapped[float] = mapped_column(Float)

    __table_args__ = (PrimaryKeyConstraint("coin", "timestamp"),)


class Content(Base):
    __tablename__ = "content_items"

    coin: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    source_id: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str] = mapped_column(String)

    __table_args__ = (PrimaryKeyConstraint("coin", "source", "content_hash"),)


class Sentiment(Base):
    __tablename__ = "sentiment"

    coin: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    content_hash: Mapped[str] = mapped_column(String)
    analyzer: Mapped[str] = mapped_column(String)
    sentiment: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        PrimaryKeyConstraint("coin", "source", "content_hash", "analyzer"),
        ForeignKeyConstraint(
            ["coin", "source", "content_hash"],
            [
                "content_items.coin",
                "content_items.source",
                "content_items.content_hash",
            ],
            ondelete="CASCADE",
        ),
    )
