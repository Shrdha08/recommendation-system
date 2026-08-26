from pydantic import BaseModel


class MovieOut(BaseModel):
    movie_idx: int
    title: str
    genres: str

    class Config:
        from_attributes = True


class RatedMovieOut(MovieOut):
    rating: float


class RecommendationOut(MovieOut):
    predicted_rating: float


class UserSummaryOut(BaseModel):
    user_count: int
    movie_count: int
