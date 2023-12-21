import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--sync-repo", default=None, help="Sync dependent git repositories to local, 'skip' for skip sync action, 'only' for only do the sync action and not launch app")
args_parser.parser.add_argument("--skip-pip", default=False, action="store_true", help="Skip automatic pip install when setup")


from args_manager import args_parser

# args_parser.parser.add_argument("--port", type=int, default=8888, help="Set the listen port, default: 8888")
args_parser.parser.add_argument("--base-url", type=str, default=None, help="Set base url for outside visit, default is http://host:port")
args_parser.parser.add_argument("--log-level", type=str, default='info', help="Log info for Uvicorn, default: info")
args_parser.parser.add_argument("--preload-pipeline", default=False, action="store_true", help="Preload pipeline before start http server")
args_parser.parser.add_argument("--queue-size", type=int, default=3, help="Working queue size, default: 3, generation requests exceeding working queue size will return failure")
args_parser.parser.add_argument("--queue-history", type=int, default=100, help="Finished jobs reserve size, tasks exceeding the limit will be deleted, including output image files, default: 100")

args_parser.parser.set_defaults(
    port=8888
)

args_parser.args = args_parser.parser.parse_args()
args = args_parser.args