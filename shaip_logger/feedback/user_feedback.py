from typing import Optional
from ..api_key import ShaipApiKey
from ..constants import API_BASE_URL
from ..request_helper import RequestHelper


class UserFeedback(ShaipApiKey):
    """
    class for logging user feedback.
    """

    @staticmethod
    def log_user_feedback(
        external_reference_id: str,
        user_feedback: int,
        user_feedback_comment: Optional[str] = None,
    ) -> None:
        """
        logs the user feedback of the prompt run given the external reference id.
        """
        try:
            payload = {
                'external_reference_id': external_reference_id,
                'user_feedback': user_feedback,
                'user_feedback_comment': user_feedback_comment,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_patch_request(
                endpoint=f'{API_BASE_URL}/api/v1/prompt_run/user-feedback',
                payload=payload,
                headers={
                    'x-api-key': UserFeedback.get_api_key(),
                },
            )
        except Exception as e:
            raise e
