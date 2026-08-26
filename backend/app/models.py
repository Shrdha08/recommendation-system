from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Movie(Base):
    __tablename__ = "movies"

    movie_idx: Mapped[int] = mapped_column(Integer, primary_key=True)  # MovieIdsNew, 0..1999
    movie_id: Mapped[int] = mapped_column(Integer, unique=True)  # original MovieLens movieId
    title: Mapped[str] = mapped_column(String, index=True)
    genres: Mapped[str] = mapped_column(String)

    ratings: Mapped[list["Rating"]] = relationship(back_populates="movie")


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_idx: Mapped[int] = mapped_column(Integer, index=True)  # 0..3373
    movie_idx: Mapped[int] = mapped_column(ForeignKey("movies.movie_idx"), index=True)
    rating: Mapped[float] = mapped_column(Float)

    movie: Mapped["Movie"] = relationship(back_populates="ratings")
