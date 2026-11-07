import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { WorkspaceProvider } from "@/components/forcecrm/WorkspaceContext";

import appCss from "../styles.css?url";
import { reportUiError } from "../lib/error-reporting";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Η σελίδα δεν βρέθηκε</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Η σελίδα που αναζητάτε δεν υπάρχει.
        </p>
        <div className="mt-6">
          <Button asChild><Link to="/">Αρχική</Link></Button>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: ErrorComponentProps) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportUiError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          Η σελίδα δεν φορτώθηκε
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {error instanceof Error ? error.message : String(error)}
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Button
            onClick={() => {
              router.invalidate();
              reset();
            }}
          >
            Επανάληψη
          </Button>
          <Button variant="outline" asChild><Link to="/">Αρχική</Link></Button>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "Τοπική ανάλυση πελατών" },
      { name: "description", content: "Τοπική ανάλυση ιδανικών πελατών και παρόμοιων επιχειρήσεων." },
      { property: "og:title", content: "Τοπική ανάλυση πελατών" },
      { property: "og:description", content: "Τοπική ανάλυση ιδανικών πελατών και παρόμοιων επιχειρήσεων." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
    links: [
      {
        rel: "stylesheet",
        href: appCss,
      },
      { rel: "icon", href: "./favicon.svg", type: "image/svg+xml" },
      { rel: "icon", href: "./favicon.ico", sizes: "any" },
      { rel: "icon", href: "./favicon.svg", type: "image/svg+xml" },
      { rel: "icon", href: "./favicon.ico", sizes: "any" },
      { rel: "icon", href: "./favicon.svg", type: "image/svg+xml" },
      { rel: "icon", href: "./favicon.ico", sizes: "any" },
      { rel: "icon", href: "./favicon.svg", type: "image/svg+xml" },
      { rel: "icon", href: "./favicon.ico", sizes: "any" },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  if (import.meta.env['VITE_STATIC_SPA'] === "true") return <>{children}</>;
  return (
    <html lang="el">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();

  return (
    <QueryClientProvider client={queryClient}>
      {import.meta.env['VITE_STATIC_SPA'] === "true" && <HeadContent />}
      {/* Required: nested routes render here. Removing <Outlet /> breaks all child routes. */}
      <WorkspaceProvider><Outlet /></WorkspaceProvider>
    </QueryClientProvider>
  );
}
