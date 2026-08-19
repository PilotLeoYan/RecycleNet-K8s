import traceback


class RecycleNetException(Exception):
    """Base exception from the RecycleNet project with enriched context."""

    def __init__(
        self,
        message: str,
        original_error: BaseException | None = None,
    ) -> None:
        self.message: str = message
        self.original_error: BaseException | None = original_error

        if original_error is not None:
            tb = original_error.__traceback__
            extracted = traceback.extract_tb(tb)
            if extracted:
                last_frame = extracted[-1]
                formatted = (
                    f"{message} | Cause: [{type(original_error).__name__}: "
                    f"{original_error}] "
                    f"en {last_frame.filename}, line {last_frame.lineno} "
                    f"({last_frame.name})"
                )
            else:
                formatted = (
                    f"{message} | Cause: [{type(original_error).__name__}: "
                    f"{original_error}]"
                )
        else:
            formatted = message

        super().__init__(formatted)
        if original_error is not None:
            self.__cause__ = original_error
