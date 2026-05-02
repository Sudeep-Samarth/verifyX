"use client"

import * as React from "react"
import { motion, HTMLMotionProps } from "framer-motion"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { login, signup } from "@/app/actions/auth"
import { useState } from "react"
import { Loader2 } from "lucide-react"

interface AuthFormProps extends HTMLMotionProps<"div"> {
  mode: "login" | "signup"
}

export function AuthForm({ className, mode, ...props }: AuthFormProps) {
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState<string | null>(null)

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    const formData = new FormData(event.currentTarget)
    
    try {
      const result = mode === "login" 
        ? await login(formData) 
        : await signup(formData)

      if (result?.error) {
        setError(result.error)
        setLoading(false)
      } else if (result && "success" in result && typeof result.success === "string") {
        setSuccess(result.success)
        setLoading(false)
      }
    } catch (err: any) {
      if (err.message !== "NEXT_REDIRECT") {
        setError("An unexpected error occurred. Please try again.")
        setLoading(false)
      }
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn("flex flex-col gap-6", className)}
      {...props}
    >
      <Card className="border-none bg-white/10 backdrop-blur-md shadow-2xl ring-1 ring-white/20">
        <CardHeader className="space-y-1">
          <CardTitle className="text-3xl font-bold tracking-tight text-white text-center">
            {mode === "login" ? "Welcome back" : "Create an account"}
          </CardTitle>
          <CardDescription className="text-white/60 text-center">
            {mode === "login"
              ? "Enter your credentials to access your chat"
              : "Enter your email to get started with Char-Chatore"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <div className="grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="email" className="text-white/80">Email</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="name@example.com"
                  required
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/30 focus-visible:ring-primary/50"
                  disabled={loading}
                />
              </div>
              <div className="grid gap-2">
                <div className="flex items-center">
                  <Label htmlFor="password" className="text-white/80">Password</Label>
                  {mode === "login" && (
                    <a
                      href="#"
                      className="ml-auto inline-block text-sm text-primary hover:underline underline-offset-4"
                    >
                      Forgot?
                    </a>
                  )}
                </div>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  required
                  className="bg-white/5 border-white/10 text-white focus-visible:ring-primary/50"
                  disabled={loading}
                />
              </div>
              
              {error && (
                <div className="text-sm font-medium text-red-500 bg-red-500/10 p-3 rounded-md border border-red-500/20">
                  {error}
                </div>
              )}

              {success && (
                <div className="text-sm font-medium text-green-400 bg-green-400/10 p-3 rounded-md border border-green-400/20">
                  {success}
                </div>
              )}

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold h-11"
              >
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {mode === "login" ? "Sign In" : "Create Account"}
              </Button>
              
              <Button
                variant="outline"
                type="button"
                className="w-full border-white/10 bg-white/5 text-white hover:bg-white/10"
                disabled={loading}
              >
                Continue with Google
              </Button>
            </div>
          </form>
          <div className="mt-6 text-center text-sm text-white/60">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <a href="/signup" className="text-primary hover:underline underline-offset-4 font-medium">
                  Sign up
                </a>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <a href="/login" className="text-primary hover:underline underline-offset-4 font-medium">
                  Log in
                </a>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
