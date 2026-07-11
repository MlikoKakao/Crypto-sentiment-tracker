--
-- PostgreSQL database dump
--

\restrict gr3mk3QdAqfc2weUu0THNXwIyUxxSnfGGATJNOWk1bbVUDIK5ZKQEnt4ZsFZ67J

-- Dumped from database version 16.14 (Debian 16.14-1.pgdg13+1)
-- Dumped by pg_dump version 16.14 (Debian 16.14-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: content_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.content_items (
    coin character varying NOT NULL,
    source character varying NOT NULL,
    source_id character varying,
    "timestamp" timestamp with time zone NOT NULL,
    text text NOT NULL,
    url character varying,
    content_hash character varying NOT NULL
);


ALTER TABLE public.content_items OWNER TO postgres;

--
-- Name: prices; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prices (
    coin character varying NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    price double precision NOT NULL
);


ALTER TABLE public.prices OWNER TO postgres;

--
-- Name: sentiment; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sentiment (
    coin character varying NOT NULL,
    source character varying NOT NULL,
    content_hash character varying NOT NULL,
    analyzer character varying NOT NULL,
    sentiment double precision NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.sentiment OWNER TO postgres;

--
-- Name: signals; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.signals (
    coin character varying NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    signal_name character varying NOT NULL,
    value double precision NOT NULL
);


ALTER TABLE public.signals OWNER TO postgres;

--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
ec85c9059503
\.


--
-- Data for Name: content_items; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.content_items (coin, source, source_id, "timestamp", text, url, content_hash) FROM stdin;
\.


--
-- Data for Name: prices; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.prices (coin, "timestamp", price) FROM stdin;
\.


--
-- Data for Name: sentiment; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sentiment (coin, source, content_hash, analyzer, sentiment, created_at) FROM stdin;
\.


--
-- Data for Name: signals; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.signals (coin, "timestamp", signal_name, value) FROM stdin;
\.


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: content_items content_items_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.content_items
    ADD CONSTRAINT content_items_pkey PRIMARY KEY (coin, source, content_hash);


--
-- Name: prices prices_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prices
    ADD CONSTRAINT prices_pkey PRIMARY KEY (coin, "timestamp");


--
-- Name: sentiment sentiment_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sentiment
    ADD CONSTRAINT sentiment_pkey PRIMARY KEY (coin, source, content_hash, analyzer);


--
-- Name: signals signals_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.signals
    ADD CONSTRAINT signals_pkey PRIMARY KEY (coin, "timestamp", signal_name);


--
-- Name: idx_content_coin_source_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_content_coin_source_id ON public.content_items USING btree (coin, source, source_id);


--
-- Name: idx_content_coin_source_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_content_coin_source_timestamp ON public.content_items USING btree (coin, source, "timestamp");


--
-- Name: idx_sentiment_coin_analyzer_source; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sentiment_coin_analyzer_source ON public.sentiment USING btree (coin, analyzer, source, content_hash);


--
-- Name: idx_signals_coin_signal_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_signals_coin_signal_timestamp ON public.signals USING btree (coin, signal_name, "timestamp");


--
-- Name: sentiment sentiment_coin_source_content_hash_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sentiment
    ADD CONSTRAINT sentiment_coin_source_content_hash_fkey FOREIGN KEY (coin, source, content_hash) REFERENCES public.content_items(coin, source, content_hash) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict gr3mk3QdAqfc2weUu0THNXwIyUxxSnfGGATJNOWk1bbVUDIK5ZKQEnt4ZsFZ67J

