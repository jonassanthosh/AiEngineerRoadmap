import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const MONTHS = [
  {
    num: 1,
    title: 'AI & ML Foundations',
    desc: 'Python, math foundations, your first neural network from scratch.',
    link: '/curriculum/month-1/what-is-ai',
  },
  {
    num: 2,
    title: 'Deep Learning',
    desc: 'CNNs, RNNs, optimizers, regularization, and transfer learning.',
    link: '/curriculum/month-2/cnns',
  },
  {
    num: 3,
    title: 'NLP & Transformers',
    desc: 'Attention mechanism, Transformer architecture from the ground up.',
    link: '/curriculum/month-3/text-preprocessing',
  },
  {
    num: 4,
    title: 'Large Language Models',
    desc: 'GPT, BERT, pretraining, tokenizers, distributed training.',
    link: '/curriculum/month-4/scaling-laws',
  },
  {
    num: 5,
    title: 'LLM Optimization',
    desc: 'LoRA, RLHF, quantization, inference optimization, deployment.',
    link: '/curriculum/month-5/fine-tuning-strategies',
  },
  {
    num: 6,
    title: 'Building New Models',
    desc: 'Read papers, design architectures, evaluate, and ship.',
    link: '/curriculum/month-6/reading-papers',
  },
];

const FEATURES = [
  {
    icon: '🧪',
    title: 'Interactive Examples',
    desc: 'Live code editors and Colab notebooks let you experiment with every concept as you learn it.',
  },
  {
    icon: '✏️',
    title: 'Hands-on Exercises',
    desc: 'Progressive exercises with hints and solutions — from beginner to advanced — at every stage.',
  },
  {
    icon: '🏗️',
    title: 'Capstone Projects',
    desc: 'Each month ends with a real project: train a model, deploy an API, write a technical report.',
  },
  {
    icon: '📚',
    title: 'Curated Resources',
    desc: 'Hand-picked papers, videos, courses, and tools linked directly from each lesson.',
  },
];

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
        <p className={styles.heroTagline}>{siteConfig.tagline}</p>
        <p className={styles.heroSub}>
          A structured, open curriculum that takes you from &ldquo;What is AI?&rdquo; to
          building and optimizing your own large language models — with code at every step.
        </p>
        <div className={styles.heroCta}>
          <Link className="button button--primary button--lg" to="/curriculum/month-1/what-is-ai">
            Start Learning
          </Link>
          <Link className="button button--outline button--lg" to="#roadmap">
            View Roadmap
          </Link>
        </div>
      </div>
    </header>
  );
}

function Roadmap() {
  return (
    <section className={styles.roadmap} id="roadmap">
      <div className="container">
        <h2 className={styles.sectionTitle}>6-Month Roadmap</h2>
        <p className={styles.sectionSub}>
          Each month builds on the last. By the end, you will have the skills to optimize existing
          models and design new architectures from scratch.
        </p>
        <div className={styles.timeline}>
          {MONTHS.map((m) => (
            <Link key={m.num} to={m.link} className={styles.timelineCard}>
              <span className={styles.monthNum}>Month {m.num}</span>
              <h3 className={styles.monthTitle}>{m.title}</h3>
              <p className={styles.monthDesc}>{m.desc}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function Features() {
  return (
    <section className={styles.features}>
      <div className="container">
        <h2 className={styles.sectionTitle}>How You Will Learn</h2>
        <div className={styles.featureGrid}>
          {FEATURES.map((f) => (
            <div key={f.title} className={styles.featureCard}>
              <span className={styles.featureIcon}>{f.icon}</span>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Home" description={siteConfig.tagline}>
      <Hero />
      <main>
        <Roadmap />
        <Features />
      </main>
    </Layout>
  );
}
