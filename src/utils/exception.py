"""Centralized exception handling for the RecycleNet project."""

import traceback


class RecycleNetException(Exception):
    """Custom base exception for the RecycleNet project with enriched error context.

    Captures the origin file, line number, and function name from the underlying
    traceback when an original exception is provided, formatting a standardized
    diagnostic message.

    Attributes:
        message: Descriptive error message explaining the failure context.
        original_error: The underlying exception that caused this error, if any.
    """

    def __init__(
        self,
        message: str,
        original_error: BaseException | None = None,
    ) -> None:
        """Initializes RecycleNetException with detailed error and traceback metadata.

        Args:
            message: High-level explanation of what failed.
            original_error: The original caught exception to chain and extract
                context from.
        """
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
                    f"in {last_frame.filename}, line {last_frame.lineno} "
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
