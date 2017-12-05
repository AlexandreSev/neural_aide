#!/usr/bin/python
# coding: utf-8

import psycopg2
import logging
import numpy as np


def create_numpy_database_from_psql(columns, save_path, limit=None,
                                    user="alex", dbname="sdss_1",
                                    rname="sdss_1_percent"):
    """
    Create a numpy file from a psql database
    Params:
        columns (list of strings): name of the columns to save.
        save_path (string): where to save the numpy file.
        limit (integer): maximum number of rows to save. If None, no limit.
        user (string): name of the user for the postgres server.
        dbname (string): name of the postgres database.
        rname (string): name of the postgres relation.
    """

    # Create a connector
    conn = psycopg2.connect("dbname=%s user=%s" % (dbname, user))
    logging.debug("Connected to postgres")

    # Create a new cursor object
    cur = conn.cursor()

    # Write the query
    query = "select %s from %s" % (", ".join(columns), rname)
    if limit is not None:
        query += " limit %s" % limit
    query += ";"
    logging.info("The query is %s" % query)

    # Execute the query
    cur.execute(query)
    logging.info("Query has been executed")

    # Fetch the results
    a = cur.fetchall()
    logging.info("Results are fetched")

    # Save the results
    np.save(save_path, a)
    logging.info("Results are saved")

    # Close the communication with the PostgresQL database
    cur.close()
    logging.debug("Connection closed")


if __name__ == "__main__":
    import argparse

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='Logging level', default="warning")
    parser.add_argument('columns', nargs='+',
                        help='Name of columns to retrieve')
    parser.add_argument('--savepath', help="Where to save the numpy",
                        default="./npy_database.npy")
    parser.add_argument('--limit', type=int, default=None,
                        help="Maximum number of rows")

    # Read the arguments
    args = parser.parse_args()

    # Verify each argument
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)

    # Configure logger
    logging.basicConfig(level=numeric_level)

    # Create the database
    create_numpy_database_from_psql(args.columns, args.savepath,
                                    limit=args.limit)
