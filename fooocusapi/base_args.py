from argparse import ArgumentParser


def add_base_args(parser: ArgumentParser, before_prepared: bool):
    if before_prepared:
        parser.add_argument("--port", type=int, default=8888, help="Set the listen port, default: 8888")
        
    parser.add_argument("--host", type=str, default='127.0.0.1', help="Set the listen host, default: 127.0.0.1")
    parser.add_argument("--base-url", type=str, default=None, help="Set base url for outside visit, default is http://host:port")
    parser.add_argument("--log-level", type=str, default='info', help="Log info for Uvicorn, default: info")
    parser.add_argument("--sync-repo", default=None, help="Sync dependent git repositories to local, 'skip' for skip sync action, 'only' for only do the sync action and not launch app")
    parser.add_argument("--skip-pip", default=False, action="store_true", help="Skip automatic pip install when setup")
    parser.add_argument("--preload-pipeline", default=False, action="store_true", help="Preload pipeline before start http server")
    parser.add_argument("--queue-size", type=int, default=3, help="Working queue size, default: 3, generation requests exceeding working queue size will return failure")
    parser.add_argument("--queue-history", type=int, default=0, help="Finished jobs reserve size, tasks exceeding the limit will be deleted, including output image files, default: 0, means no limit")
    parser.add_argument('--webhook-url', type=str, default=None, help='The URL to send a POST request when a job is finished')