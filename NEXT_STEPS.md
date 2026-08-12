Current architecture:
    Project is fully deployed on Rpi with ingest every 3 hours.
    The structure of project is on AWS without data, without automated ingest. Just pure architecture

Goal architecture:
    Deployed on AWS with Rpi architecture supplied
    https
    DNS, not just raw IP
    Possible rewrite to juts show all DB data instead of querying it? Seems to put less pressure on the DB if done like this, also makes more sense for the UI.
