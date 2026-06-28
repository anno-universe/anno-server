from ninja_extra import NinjaExtraAPI
from ninja_jwt.controller import NinjaJWTDefaultController

api = NinjaExtraAPI(
    title="Anno API",
    version="0.1.0",
)

api.register_controllers(NinjaJWTDefaultController)
api.auto_discover_controllers()
