"""
Qdrant Management Script

This script provides utilities for managing Qdrant collections, including
listing, cleaning, and deleting collections.
"""

import argparse
from typing import List, Optional
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm
from dotenv import load_dotenv
from utils.logging_config import setup_logging
from utils.common import get_qdrant_client

# Set up logger
logger = setup_logging("qdrant_manager")


def list_collections(client) -> List[str]:
    """
    List all available collections.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance

    Returns
    -------
    List[str]
        List of collection names
    """
    try:
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except UnexpectedResponse as e:
        logger.error(f"Error listing collections: {e}")
        return []


def get_collection_info(client, collection_name: str) -> dict:
    """
    Get detailed information about a collection.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance
    collection_name : str
        Name of the collection

    Returns
    -------
    dict
        Collection information
    """
    try:
        info = client.get_collection(collection_name)
        points_count = client.count(collection_name).count
        return {
            "name": collection_name,
            "vector_size": info.config.params.vectors.size,
            "points_count": points_count,
            "status": "green" if info.status == "green" else "not ready"
        }
    except UnexpectedResponse as e:
        logger.error(f"Error getting info for collection {collection_name}: {e}")
        return {}


def delete_collection(client, collection_name: str):
    """
    Delete a specific collection.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance
    collection_name : str
        Name of the collection to delete
    """
    try:
        client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully")
    except UnexpectedResponse as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")


def delete_all_collections(client, exclude: Optional[List[str]] = None):
    """
    Delete all collections except excluded ones.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance
    exclude : Optional[List[str]]
        List of collection names to exclude from deletion
    """
    exclude = exclude or []
    collections = list_collections(client)

    for collection in tqdm(collections, desc="Deleting collections"):
        if collection not in exclude:
            delete_collection(client, collection)


def main():
    parser = argparse.ArgumentParser(description='Manage Qdrant collections')
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local Qdrant instance (default is cloud)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser('list', help='List all collections')
    list_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information about each collection'
    )

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete collections')
    delete_parser.add_argument(
        '--collections',
        nargs='+',
        help='Names of collections to delete (default: all)'
    )
    delete_parser.add_argument(
        '--exclude',
        nargs='+',
        help='Names of collections to exclude from deletion'
    )
    delete_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    try:
        # Initialize Qdrant client
        client = get_qdrant_client(local=args.local)

        if args.command == 'list':
            collections = list_collections(client)
            if not collections:
                logger.info("No collections found")
                return

            if args.detailed:
                logger.info("\nCollection Details:")
                print("-" * 70)
                for collection in collections:
                    info = get_collection_info(client, collection)
                    print(f"\nCollection: {info['name']}")
                    print(f"Vector Size: {info['vector_size']}")
                    print(f"Points Count: {info['points_count']}")
                    print(f"Status: {info['status']}")
                    print("-" * 70)
            else:
                logger.info("\nAvailable collections:")
                for collection in collections:
                    print(f"- {collection}")

        elif args.command == 'delete':
            if args.collections:
                # Delete specific collections
                if not args.force:
                    confirm = input(
                        f"Are you sure you want to delete these collections: {', '.join(args.collections)}? [y/N] ")
                    if confirm.lower() != 'y':
                        logger.info("Operation cancelled")
                        return

                for collection in args.collections:
                    delete_collection(client, collection)
            else:
                # Delete all collections except excluded
                if not args.force:
                    confirm = input("Are you sure you want to delete ALL collections? [y/N] ")
                    if confirm.lower() != 'y':
                        logger.info("Operation cancelled")
                        return

                delete_all_collections(client, exclude=args.exclude)

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
