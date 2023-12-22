from fooocusapi.base_args import add_base_args
import ldm_patched.modules.args_parser as args_parser

# Add Fooocus-API args to parser
add_base_args(args_parser.parser, False)

# Apply Fooocus's args
from args_manager import args_parser

# Override the port default value
args_parser.parser.set_defaults(
    port=8888
)

# Execute args parse again
args_parser.args = args_parser.parser.parse_args()
args = args_parser.args